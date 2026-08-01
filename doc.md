# Dynamic Experts v2 — Architecture Redesign

## 0. Why v1 Failed

### 0.1 Symptom

Model produces garbled output, NaN by layer 6 of prefill. Gate/up/down weight data verified byte-perfect. Slot_map verified correct on GPU. Bias tensors nulled out. Still garbage.

### 0.2 Root Cause (Hypothesized)

We fought ggml's buffer system for 40+ hours. We used `ggml_backend_buft_alloc_buffer` for slot buffers, which required:

- EXTERNAL flags on every slot tensor and slot_map tensor
- Buffer save/restore in `process_ubatch` to prevent scheduler overwrite
- `supports_buft` hacks to make slave CRASH accept our buffers
- `raw_buf_write` with static CUDA streams and manual sync

This created a fragile system where the scheduler could corrupt our pointers at any graph allocation. We verified it didn't (slot buffer pointers unchanged), but the `ggml_get_rows` remap output might have been corrupted between groups.

Additionally, we tried bypassing vec_q/MMQ/MMF kernels because we assumed `ne02=16` broke them. We added checks in 5+ different places (direct path, fusion path, should_use_mmq, mmq.cu early return, mmf.cu early return). This was wrong: the kernels should handle `ne02=16` correctly — the NaN was from our ggml buffer corruption, not from the kernels.

### 0.3 The Flash-MOE Insight

The reference implementation at `llama.cpp-flash-moe` (which works, just slower) does something fundamentally different: it **bypasses ggml's buffer system entirely**.

Instead of:
```
ggml_backend_buft_alloc_buffer → raw_buf_write → scheduler fights
```

It does:
```
cudaMalloc → cudaMemcpyAsync → tensor.data points to cudaMalloc'd memory
```

No ggml buffers involved for expert weights. The `ggml_tensor` has `data` pointing to raw `cudaMalloc` memory. The tensor lives outside the ggml context (heap-allocated with `new`). The scheduler doesn't know about this memory and doesn't touch it.

## 1. New Architecture

### 1.1 Memory Management

**Slot buffer**: one `cudaMalloc` block per layer:
```
d_slots[layer] = cudaMalloc(n_slots * expert_total_size)
```

Where `expert_total_size = gate_size + up_size + down_size` (all three weight types packed into one block per slot). Or separate buffers per weight type — whichever is cleaner.

**CPU staging buffer**: one `malloc` block, reused per load:
```
host_staging = malloc(max_expert_size)
```

**Slot map (GPU)**: one `cudaMalloc` block total:
```
d_slot_map = cudaMalloc(n_layers * n_experts * sizeof(int32_t))
```

**Slot map (host)**: mirrored in `std::vector`:
```
h_slot_of[n_layers * n_experts]  // expert -> slot
h_expert_in[n_layers * n_slots]  // slot -> expert
```

### 1.2 Slot Tensor Creation

After `load_arch_tensors`, for each layer:
1. Create a `new ggml_tensor` (heap, not in ggml context)
2. Set `ne = [ne0, ne1, n_slots]`, `nb` computed from type
3. Set `data = d_slots[layer] + weight_type_offset`
4. Set `buffer = nullptr` (scheduler sees nullptr, allocates nothing)
5. Set `flags = GGML_TENSOR_FLAG_EXTERNAL`
6. Replace `layer.ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` with this tensor

The scheduler sees `buffer == nullptr` → skips allocation. The `data` pointer is valid GPU memory. The kernel reads from it.

### 1.3 Loading an Expert

```
dyn_ex_cache_load(layer, expert_id, slot_idx):
    for each weight_type (gate, up, down):
        offset = dyn_ex_read_param(reader, pi, layer, expert_id)  // .bin file offset
        memcpy from mmap'd .bin at offset into host_staging
        dst = d_slots[layer] + slot_idx * expert_total_size + type_offset
        cudaMemcpyAsync(dst, host_staging, expert_size, H2D, stream)
    h_slot_of[layer*nexperts + expert_id] = slot_idx
    h_expert_in[layer*nslots + slot_idx] = expert_id
    cudaMemcpyAsync(d_slot_map + layer*nexperts + expert_id, &slot_idx, 4, H2D, stream)
```

### 1.4 Slot Tensor Layout (Flat Packing)

All three weight types packed into one contiguous block per slot:

```
Slot 0: [gate_bytes | up_bytes | down_bytes]
Slot 1: [gate_bytes | up_bytes | down_bytes]
...
```

Each `ggml_tensor` for gate/up/down points to the correct offset within this block:
```
gate_tensor.data = d_slots[layer] + slot * total_stride + 0
up_tensor.data   = d_slots[layer] + slot * total_stride + gate_size
down_tensor.data = d_slots[layer] + slot * total_stride + gate_size + up_size
```

### 1.5 Barrier / Synchronization

Keep the eval callback approach (working):

1. `DYN_EX_BARRIER` node in graph splits Group A (router) from Group B (matmul)
2. Eval callback fires between groups
3. Callback reads `selected_experts` from `t->src[0]` (router output)
4. Calls `dyn_ex_cache_ensure_ordered(layer, expert_ids, n_ids)` to load experts
5. No thread needed — callback runs on main thread synchronously

**Why this works now**: with cudaMalloc'd memory, there's no risk of the scheduler modifying our buffer pointers between groups. The callback can freely read/write our memory without interference.

### 1.6 Remap (Unchanged)

Keep `ggml_get_rows(slot_map[layer], selected_experts)` in `build_moe_ffn`. This is proven correct: produces [0..7] for all experts loaded by ensure_ordered. The `slot_map` tensor uses `d_slot_map` as its data pointer.

### 1.7 Kernel Paths (Unchanged)

**ZERO kernel modifications**. The slot tensor has `ne02 = n_slots` just like before. Let the stock `ggml_cuda_mul_mat_id` dispatch to whatever kernel it wants (vec_q, MMQ, sorted). If any kernel truly has a bug with `ne02=16`, that's a llama.cpp bug to report, not our problem.

The flash-moe implementation works because it uses FP16 weights and the sorted path handles them correctly. We'll use Q4_K/Q6_K and accept whatever path the dispatcher chooses. If it doesn't produce correct output, we'll use a larger `n_slots` or report the kernel bug.

### 1.8 3-Tier Caching

**L1**: GPU slots (`d_slots` via cudaMalloc)
**L2**: Host RAM (`host_staging` via malloc, reused)
**L3**: Disk (mmap'd .bin file, OS page cache)

Flow: L3 (mmap read) → L2 (host_staging) → L1 (cudaMemcpyAsync to d_slots)

## 2. What We Tried and Why It Failed

### 2.1 ggml_buft_alloc_buffer + EXTERNAL

**Problem**: The scheduler in `ggml_backend_sched_alloc_graph` might reallocate or copy tensors. EXTERNAL flag prevents copying but causes `supports_buft` issues with CUDA.

**Failure**: Constant fighting with ggml's buffer management. Buffer pointers needed save/restore, and `raw_buf_write` sync timing was fragile.

### 2.2 Kernel Bypass (vec_q, MMQ, MMF)

**Problem**: Assumed `ne02=16` broke optimized kernels. Added early returns and should_use checks to force sorted path.

**Failure**: The NaN was not from kernels — it was from ggml buffer corruption. The bypasses masked the real issue and made us go in circles for days. Also, the `return` from MMQ/vec_q/MMF didn't fall through properly because callers always `return` after calling these functions.

### 2.3 Spinning Barrier Thread

**Problem**: Needed synchronization between GPU router output and CPU expert loading. Used `cudaHostAllocMapped` + GPU spin kernel + CPU thread.

**Failure**: Worked for single token but deadlocked with multi-token. Also fought with the scheduler over barrier buffer ownership.

### 2.4 Two-Pass Inference

**Problem**: Tried to capture expert IDs in pass 1, load weights, then run pass 2.

**Failure**: Inherently flawed — same layer computed twice, compute waste, and the hidden state diverges between passes.

### 2.5 CPU Baseline Comparison

**Problem**: Needed to verify dyn-ex output matches original.

**Failure**: 35B model doesn't fit in RAM for CPU inference. Segfault on CPU path suggests our code broke the no-dyn-ex path. Never resolved.

## 3. Implementation Plan

### Phase 1: Memory Backend Swap

Replace `ggml_backend_buft_alloc_buffer` / `raw_buf_write` with `cudaMalloc` / `cudaMemcpyAsync`:

1. Create `dyn_ex_slot_buffer` struct with `cudaMalloc`'d pointers
2. Rewrite `dyn_ex_cache_init` to use cudaMalloc
3. Rewrite `dyn_ex_cache_ensure` to use cudaMemcpyAsync
4. Update `make_slot_tensor` to point `data` at cudaMalloc'd memory
5. Remove all `raw_buf_write` calls
6. Remove `buffer` save/restore in `process_ubatch`

### Phase 2: Slot Map

Replace ggml buffer slot_map with cudaMalloc:
1. `d_slot_map = cudaMalloc(n_layers * n_experts * sizeof(int32_t))`
2. Tensor `data` points to `d_slot_map + layer_offset`
3. Sync from `h_slot_of` via `cudaMemcpyAsync`

### Phase 3: Test

1. Build and test with `--dyn-ex-l1 16`
2. If output is garbled: try `--dyn-ex-l1 64`
3. If still garbled: try `--dyn-ex-l1 256` (full size, should match original)
4. If 256 slots works but 16 doesn't: kernel issue with ne02=16
5. If 256 slots also fails: loading/data issue

### Phase 4: Performance

1. Add CUDA stream for async copies
2. Add CUDA events for synchronization
3. Add L2 host cache (optional, after L1 works)

## 4. Files to Modify

- `src/llama-dyn-ex.h`: new slot buffer struct, remove `buf_gate`/`buf_up`/`buf_down`
- `src/llama-dyn-ex.cpp`: cudaMalloc, cudaMemcpyAsync, remove raw_buf_write
- `src/llama-model.cpp`: update `make_slot_tensor`, remove buffer dependency
- `src/llama-context.cpp`: remove buffer save/restore, keep callback
- `src/llama-model-loader.cpp`: keep expert skip (unchanged)
- `src/llama-graph.cpp`: keep remap (unchanged)

## 5. Files to DELETE

- All ggml/ kernel modifications (revert to stock)
- `raw_buf_write` helper
- `dyn_ex_cache_alloc_barriers` (unused with callback)
- All the vec_q/MMQ/MMF bypass code
