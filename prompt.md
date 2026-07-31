# Dynamic Experts for llama.cpp

## Overview

Slot-based dynamic expert loading for MoE LLMs in llama.cpp. An MLP predictor tries to guess which experts from the hidden state of layer n at token t will be needed at token t+1. This is an **optimization**, not the expert decider — the main LLM's router still makes the final decision.

Terminology:
- **MLP/predictor**: The small model that predicts next-token expert selection
- **LLM**: The main MoE model
- **V6Model**: The Python reference predictor (from vllm-import)

---

## Cache Architecture (DECIDED: 2-level)

```
L1: GPU slots in VRAM (n_slots per layer, statically allocated, contiguous)
L2: mmap'd byte-aligned .bin file (OS page cache = free RAM caching, disk = cold miss)
```

**Decision**: Drop explicit L2 (host RAM buffer). The mmap'd .bin file gets cached in the OS page cache automatically. Sequential access from prefetching keeps hot experts in RAM. Cold misses are rare after warmup. This matches the vllm-import approach where the mmap reader goes directly from disk/page-cache to GPU via async copy.

Cache loading: L2 (disk/mmap) → L1 (GPU). No intermediate host buffer.

---

## Slot & Slot Map Design

```
slots[L][n_slots]      -> 2D array, each layer has n_slots expert slots
slot_map[L][n_expert]  -> given (layer, expert_id), returns slot index (or -1 if not present)
expert_in[L][n_slots]  -> given (layer, slot_idx), returns expert_id (or -1 if empty)
slot_in_use[L][n_slots] -> bool, whether slot is occupied
```

**Why slot_map?** The LLM router returns expert IDs 0..n_expert-1, but we only have n_slots slots. The slot_map remaps: `slot_of(layer, expert_id)` → slot index. Before the graph op, all expert IDs are remapped to slot-local indices (0..n_slots-1).

---

## Operations

### get_experts(layer, expert_ids)
Used on the LLM side. Single source of experts. Checks L1: if expert is present in a slot, returns the slot index. If not, evicts the lowest-utility expert and loads from the mmap'd .bin file into the freed slot. **Always called, regardless of whether the MLP made a prediction.**

### prefetch_experts(layer, expert_ids, scores)
Used on the MLP predictor side. After prediction, loads predicted experts into L1 via async copies (separate CUDA stream). Non-blocking — LLM continues on the default stream.

### evict(layer, scores)
When a slot is needed and all are full, evicts the expert with the lowest utility score. Score comes from the predictor's logits. If no scores available, falls back to FIFO.

### fill_slots()
Loads the first n_slots experts into slots during initialization. Ensures at least n_slots unique experts are resident before first inference.

---

## MLP Predictor

Train: on CPU, after decode finishes and d_trace is offloaded to h_trace.
Predict: on GPU, using hidden state of layer n at token t to forecast experts for token t+1.
load_weights: before decode (loads trained weights from disk).
save_weights: after train finishes.

Architecture (from vllm-import V6Model):
- Input: hidden_state (batch, D) + current expert mask (batch, E) + layer index
- trunk: Linear(D, 16) → ReLU
- Per-layer: Concat(z, Et) → Linear(16+E, 16) → ReLU → Linear(16, E)
- Output: logits over all experts (batch, E)
- Training: self-supervised BCE loss (predict what the router chose)
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4

---

## CLI Args (Planned)

```
--dyn-ex <path>           -> enables dynamic experts, path to byte-aligned .bin file
--dyn-ex-predictor <path> -> MLP predictor weights. Dir = create {model}.mlp after first train.
--dyn-ex-l1 <num>         -> number of expert slots per layer in VRAM (default: 32)
--dyn-ex-l2 <num>         -> DROPPED (see cache architecture decision above)
--dyn-ex-trace <dir>      -> whether to gather full trace for offline analysis
```

---

## Integration Points (llama.cpp)

### 1. Model Loading — Skip Expert Weight VRAM Allocation
**File**: `src/llama-model-loader.cpp` — `load_all_data()` (line 1402)
**File**: `src/llama-model.cpp` — `llama_model_base::load_tensors()` (line 1253)

Pattern: Expert tensors are identified by `_exps` suffix in tensor name. During `load_all_data()`, skip backend buffer allocation and data load for any tensor matching `.*_exps.*`. Instead, mmap the pre-built .bin file and store a reader for on-demand access.

Existing precedent: `--cpu-moe` flag (arg.cpp line 2627) already identifies experts via `_exps` regex for tensor buft overrides.

### 2. Graph Building — Expert Remapping in build_moe_ffn
**File**: `src/llama-graph.cpp` — `build_moe_ffn()` (line 1766-2158)
**File**: `src/llama-graph.h` — declarations (line 1001-1046)

Key finding: `selected_experts_in` parameter (line 1924) allows passing pre-computed expert indices — no caller currently uses this. This is THE hook: pass slot-remapped expert IDs instead of true expert IDs.

Flow:
1. get_experts(layer, router_topk_ids) → ensures experts are in slots
2. slot_of(router_topk_ids) → remaps to slot indices (0..n_slots-1)
3. Pass slot indices as `selected_experts_in` to `build_moe_ffn()`
4. Graph's `ggml_mul_mat_id` uses slot indices as if they were expert indices
5. Expert weight tensor has shape `[in, out, n_slots]` (NOT `[n_in, out, n_expert]`)

### 3. Graph Replay — Static Graph with Dynamic Slot Data
**File**: `src/llama-context.cpp` — `process_ubatch()` (line 1317)
**File**: `src/llama-graph.cpp` — `can_reuse()` (line 1274)

The graph reuse check compares tensor SHAPES, not data contents. Expert weight tensor has fixed shape `[in, out, n_slots]`. When new experts are loaded into slots, the tensor data changes but the shape doesn't → graph reuses fine.

Data flow during replay:
1. `can_reuse()` → true (shapes unchanged)
2. `set_inputs()` → updates input tensor data (standard)
3. `ggml_backend_tensor_set(expert_tensor, slot_data, ...)` → or the tensor IS the slot buffer
4. `graph_compute()` → runs with new slot data

### 4. The Callback / Hook Point
**File**: `src/llama-graph.h` — `llm_graph_cb` (line 669)

The `llm_graph_cb` callback is invoked for every tensor during graph building. This is where we could insert hooks for trace capture (when `--dyn-ex-trace` is enabled) and for predictor data collection.

### 5. Prefetch — Async on Separate Stream
Use a dedicated CUDA stream (separate from the default compute stream) for H2D copies during prefetch. This keeps PCIe usage for expert loading decoupled from LLM compute. Pattern from vllm-import: `torch.cuda.Stream` → CUDA stream + events for synchronization.

### 6. CLI Arg Registration
**File**: `common/common.h` — `struct common_params` (~line 550)
**File**: `common/arg.cpp` — `common_params_parser_init()` (line 1317+)
**File**: `common/common.cpp` — `common_model_params_to_llama()` (line 1576)

Follow the existing pattern: add fields to `common_params`, register args with `add_opt(common_arg(...))`, wire into model params conversion.

---

## Binary File Format (VLLM\x02 / v2)

Pre-built using `vllm-import/convert-gguf-to-expert-binary.py`:
```
python convert-gguf-to-expert-binary.py model.gguf expert_weights.bin
```

Layout:
```
[Header: 16384 bytes, page-aligned]
  magic:    "VLLM\x02" (6 bytes)
  version:  uint32_le   (4 bytes) = 2
  n_layers: uint32_le   (4 bytes)
  n_experts: uint32_le  (4 bytes)
  expert_stride: uint64_le (8 bytes)
  n_params: uint16_le   (2 bytes)
  per-param: name_len(1) + name + ndim(1) + shape(ndim*4) + dtype_code(1)
  reserved: pad to 16384

[Data: L * E blocks, each expert_stride bytes, 4KB aligned]
  offset = 16384 + (layer * n_experts + expert) * expert_stride
  each block: param0_data | param1_data | ... | pad to page boundary
```

O(1) lookup — no index table needed. Reading in C++: mmap the file, parse header, seek + read per expert.

---

## Edge Cases & Open Questions

### Resolved

- **Stop llama.cpp from loading expert weights into VRAM**: Skip in `load_all_data()` line 1517. Match expert tensor names with `strstr(name, "_exps")` — no regex. Expert tensors stay in the ggml context (for GGUF metadata compatibility) but get `data = nullptr`, `buffer = nullptr` — never allocated. Non-expert weights load as usual. Works cleanly on the mmap path where each tensor is allocated individually via `ggml_backend_tensor_alloc()`. On the non-mmap path (`ggml_backend_alloc_ctx_tensors_from_buft` allocates all tensors in a context at once), expert tensors would need to be placed in a separate context. Primary path is mmap (the .bin is always mmap'd).
- **Load only static (non-expert) weights at startup**: Same mechanism — skip `_exps` tensors.
- **Graph replay with dynamic data**: Since `can_reuse()` checks shapes not data, and `ggml_mul_mat_id` reads from tensor data at compute time, the graph works with slot-local data as long as the tensor shape stays `[in, out, n_slots]` across replays.
- **L2 dropped**: mmap'd .bin gets OS page cache as free L2. Always mmap the .bin file.
- **Binary format**: Reuse VLLM\x02 v2 format. Pre-built offline.
- **Tensor shape for ggml_mul_mat_id**: The `as` tensor in `ggml_mul_mat_id(as, b, ids)` is sized `[in, out, n_slots]`. It IS the slot cache buffer — shared across all graph replays, fixed size, only contents change. Expert weights are loaded into slots before compute.
- **Expert ID → slot index remapping**: Insert `ggml_get_rows(slot_map[layer], expert_ids)` as a graph op between the router's `ggml_argsort_top_k` and `ggml_mul_mat_id`. The `slot_map` is `[n_layers, n_expert]` int32 with `SENTINEL` fill. `get_experts()` updates `slot_map` when loading experts into slots. Remapping happens at compute time as part of the graph.
- **Slot cache structure**: `[n_layers][n_slots]` of struct `{ up: float*, gate: float*, down: float* }` — three weight buffers per slot. Shapes determined by model architecture (n_embd, n_ff). The GPU buffer backing these pointers is the same `[in, out, n_slots]` tensor used by `ggml_mul_mat_id`.
- **--cpu-moe does NOT skip loading**: Verified in code — `create_tensor()` → `buft_for_tensor()` only overrides buffer TYPE to CPU, but data is still loaded. `load_all_data()` allocates and copies data regardless. This wastes RAM. Our approach skips allocation entirely via `continue` in the load loop.
- **Expert tensor name matching**: Use `strstr(name, "_exps")` — no regex. The `_exps` suffix is reliable: `ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`, `ffn_gate_up_exps`. Non-expert tensors are `ffn_gate`, `ffn_down`, `ffn_up` (no `_exps` suffix).

### Still Open
(None — all answered below.)

### Decided
- **Where to store the slot cache struct**: Global-ish, long-lived. Possibly a static/global pointer or owned by `llama_model`. Must survive multiple forward passes and graph replays. Cleaned up in destructor.
- **ggml_mul_mat_id with arbitrary n_slots**: Enforce n_slots is power of 2. If user passes non-power-of-2, just let it crash — won't fix.
- **Monolithic vs modular kernel fallback**: Ignore for now. Will revisit if problems arise.
- **First token penalty**: Accepted. get_experts() does blocking load for first token. No prediction available at t=0.
- **Async prediction and prefetch**: Predict on GPU using hidden state from `process_ubatch`. Prefetch async on separate CUDA stream, running in parallel with attention/FFN of subsequent layers. Predictor must not stall LLM.
- **Minimal PCIe usage**: Only expert loading uses PCIe during active inference. Predictor runs on GPU (no PCIe). Expert async copies on separate stream, scheduled during idle PCIe windows.

---

## Code Separation Convention

All dynamic experts code should be clearly marked with `// dyn-ex:` prefix. New files should go in `src/dyn-ex/` or use `llama-dyn-ex-*` naming.

Prefer conditional compilation (`#ifdef LLAMA_DYN_EX`) or runtime feature flags over code duplication.

---

## Exploration Log

### 2026-07-29 — Codebase familiarization
- Read entire vllm-import/ directory (14 files): understands the Python reference architecture
- Read llama.cpp integration points: build_moe_ffn, llama_model_loader, llama_layer, process_ubatch, ggml_mul_mat_id, graph reuse
- Identified `selected_experts_in` as the cleanest integration hook (unused by any caller)
- Confirmed `can_reuse()` checks topology only (shapes), not data — enabling dynamic slot data
- Confirmed `_exps` suffix as reliable expert tensor discriminator (used by existing `--cpu-moe`)
- Decided 2-level cache (L1 GPU + mmap), dropping explicit L2 host buffer
- Decided to reuse VLLM\x02 binary format (pre-built offline via convert-gguf-to-expert-binary.py)

### 2026-07-29 — Design decisions
- Traced `load_all_data()` (llama-model-loader.cpp:1402-1684): mmap path allocates tensors individually via `ggml_backend_tensor_alloc()` (line 1548) — can skip per-tensor with `strstr(name, "_exps")` → `continue`
- Verified `--cpu-moe`: does NOT skip loading, only changes buffer type to CPU. RAM is wasted.
- Traced `create_tensor()` / `buft_for_tensor()`: expert tensor buffer type selection flow
- Decided on `ggml_get_rows(slot_map, expert_ids)` remap op between routing and `ggml_mul_mat_id` — expert IDs remapped to slot indices at compute time
- Slot cache struct: `[n_layers][n_slots]` of struct `{up, gate, down}` — three weight buffers per slot, shapes determined by model architecture. Allocated in contiguous GPU buffers (one per weight type: gate_up, down).
- Always mmap the .bin file. Expert tensors use `n_slots` shape for the graph op.
- Implementation order: (1) load_all_data skip, (2) slot cache + .bin reader, (3) ggml_get_rows remap, (4) get_experts/ensure, (5) CLI args, (6) perfetch, (7) predictor

### 2026-07-29 — Implementation progress

#### Step 1: load_all_data skip ✓
**File**: `src/llama-model-loader.cpp` line ~1522
- Added `strstr(tname, "_exps.") || strstr(tname, "_exps_")` check after weight lookup
- Skipped tensors increment `size_done` for accurate progress tracking
- Expert tensors remain in ggml context but get `data = nullptr`, `buffer = nullptr`

#### Step 2: .bin reader + cmake ✓
**Files**: `src/llama-dyn-ex.h`, `src/llama-dyn-ex.cpp`, `src/CMakeLists.txt`
- `dyn_ex_reader`: mmaps .bin, parses VLLM\x02 v2 header, O(1) per-expert read
- Converter script is source of truth (per-param layout, GGML_TYPE_TO_DTYPE_CODE mapping)
- `dyn_ex_param_size()`, `dyn_ex_param_index()`, `dyn_ex_read_param()` helpers

#### Step 3: ggml_get_rows remap in build_moe_ffn ✓
**Files**: `src/llama-graph.h`, `src/llama-graph.cpp`
- Added `ggml_tensor * slot_map = nullptr` parameter to both `build_moe_ffn` overloads
- After `selected_experts` computed: `ggml_get_rows(slot_map, selected_experts)` → `selected_experts_slots`
- 4 `build_lora_mm_id` calls use `selected_experts_slots` (slot indices for matmul dispatch)
- 3 `ggml_add_id` (bias) calls keep `selected_experts` (true expert IDs — biases stored at full size)
- Backward compatible: `slot_map == nullptr` → `selected_experts_slots = selected_experts` (no-op)

#### Step 4: slot cache + ensure ✓
**Files**: `src/llama-dyn-ex.h`, `src/llama-dyn-ex.cpp`
- `dyn_ex_cache` struct: GPU buffers for gate_up slots, down slots, slot_map; host-side tracking arrays
- `dyn_ex_cache_init()`: creates GPU buffers (aligned), validates params, maps param names
- `dyn_ex_cache_ensure()`: loads experts from .bin → CPU staging buffer → GPU slot (blocking)
  - Checks if already loaded (h_slot_of lookup), round-robin free-slot allocation
  - Evicts if all slots occupied, syncs slot_map to GPU after changes
- `dyn_ex_cache_fill()`: loads experts 0..n_slots-1 per layer at init
- Handles merged gate_up path; separate gate/up path marked TODO

#### Step 5: CLI args + model loading integration ✓
**Files**: `include/llama.h`, `common/common.h`, `common/arg.cpp`, `common/common.cpp`, `src/llama-model.h`, `src/llama-model.cpp`
- Added `--dyn-ex <FILE>`, `--dyn-ex-l1 <N>`, `--dyn-ex-predictor <FILE>` to arg parser
- Added fields to `llama_model_params` and `common_params`
- Wired `common_params` → `llama_model_params` in `common_model_params_to_llama()`
- Added `ffn_slot_map` (int32, n_expert elements) to `llama_layer`
- In `load_tensors()`: after `load_arch_tensors(ml)`, opens .bin, creates cache, replaces per-layer expert tensor pointers with slot-backed tensors (raw ggml_tensor structs allocated in cache buffers)
- `make_slot_tensor()` helper: creates ggml_tensor with `[ne0, ne1, n_slots]` shape, allocated in cache GPU buffer at correct offset
- `dyn_ex_cache_fill()` called at end to preload first n_slots experts
- Destructor: frees cache, per-layer tensors, slot_map tensors

#### Step 6: Per-model slot_map plumbing (TODO)
47 model files call `build_moe_ffn`. Each call site needs `layer.ffn_slot_map` passed as the last argument. Mechanical change — infrastructure supports it (defaults to nullptr, backward compatible).

#### Files created:
- `src/llama-dyn-ex.h` — reader + cache API
- `src/llama-dyn-ex.cpp` — reader + cache implementation (~320 lines)

#### Files modified:
- `src/llama-model-loader.cpp` — expert tensor skip
- `src/llama-graph.h` — `slot_map` parameter
- `src/llama-graph.cpp` — remap op + slot dispatch
- `src/llama-model.h` — `ffn_slot_map` field
- `src/llama-model.cpp` — cache init, cleanup, include
- `src/CMakeLists.txt` — add `llama-dyn-ex.cpp`
- `include/llama.h` — model params
- `common/common.h` — common params
- `common/arg.cpp` — CLI args
- `common/common.cpp` — params wiring

---

## Current Status (2026-07-30)

### Working
- **Model loading**: expert weight tensors skipped in `load_all_data()` via `strstr(name, "_exps.")` check
- **.bin reader**: VLLM\x02 format reader/writer, converter script
- **Slot cache**: GPU buffers created per weight type (gate_up, gate, up, down), per-layer slot tensors replace original `llama_layer` expert pointers
- **Slot data verified byte-per-byte**: `dyn_ex_cache_fill` loads correct expert weights from .bin into slots
- **Slot tensor buffers confirmed unchanged**: scheduler does NOT reallocate pre-allocated slot buffers during `sched_alloc_graph`
- **Async prefetch infra**: CUDA stream, events, staging buffers, `prefetch()` / `wait()` functions
- **CPU predictor**: V6Model architecture, AdamW training, DXP2 binary format save/load
- **CLI args**: `--dyn-ex`, `--dyn-ex-l1`, `--dyn-ex-predictor` wired through to model params
- **All 47 model archs**: `ffn_slot_map` passed to `build_moe_ffn`
- **GGML_OP_DYN_EX_BARRIER**: custom ggml op with CUDA spin-wait kernel + CPU memcpy handler
- **Scheduler group-break**: `ggml-backend.cpp` breaks compute groups at `DYN_EX_BARRIER` nodes
- **Model loads and runs inference**: without crash on basic path

### BLOCKING: NaN in softmax during inference
- NaN assertion fires in `ggml_compute_forward_soft_max_f32` (CPU backend)
- Happens during **attention softmax**, BEFORE any MoE computation
- Model works without `--dyn-ex` (confirmed) but doesn't fit in VRAM (18GB needed)
- With `--dyn-ex`, expert weights skipped → model fits, but NaN in attention
- **Hypothesis**: `load_all_data()` skip might accidentally skip non-expert tensors (attention weights, embeddings, norms, etc.) — the `strstr` pattern `"_exps."` should only match MoE expert tensors, but needs verification

### What's NOT the issue
- **Scheduler buffer reallocation**: Confirmed slot tensor buffers unchanged after `sched_alloc_graph`
- **Expert data corruption**: Verified byte-per-byte match between .bin source and slot tensor
- **Barrier/busy-wait approach**: Thread starts and waits for ready flag — confirmed working
- **Eval callback approach**: Fires but barrier nodes don't reach compute splits (scheduler grouping issue)

### Architecture decisions
- **Barrier thread + CUDA spin-wait**: chosen over eval callback (simpler, no scheduler fighting)
- **ggml_op_is_empty**: DYN_EX_BARRIER REMOVED from empty ops — dispatches to all backends
- **Barrier buffer type**: Device buffer (not host-visible) — matches scheduler split type
- **Barrier read**: Thread reads from `res->dyn_ex_barrier[il]` (graph copy, not cache copy)

---

## Current Status (2026-07-31)

### Working
- **Model loading**: TENSOR_SKIP prevents expert weight GPU allocation
- **.bin reader**: VLLM\x02 format, O(1) per-expert reads
- **Slot cache**: GPU buffers per weight type, correct Q4_K strides
- **Barrier mechanism**: cudaHostAllocMapped + CPU thread + GPU kernel, verified working (L0 go set)
- **Slot_map**: raw_buf_write updates GPU (cudaMemcpyAsync on non-blocking stream)
- **Remap**: ggml_get_rows(slot_map, flat) → reshape → cont, produces correct slot indices [0..15]
- **Barrier thread**: polls ready via host_ptr, loads experts via ensure(), sets go
- **Ensure()**: loads expert weights from .bin, smart eviction (skips needed experts)
- **can_seq_rm skip**: llama_model_has_dyn_ex() API, skips 2-token test during server init

### Failing
- **Gate matmul crashes**: ggml_cuda_mul_mat_id for gate_exps has illegal memory access
  - Matmul launches ok (cudaErr: ok) but kernel crashes async
  - `ids_dst=0x1` in scatter path — corrupt pool pointer
  - With `dedup_bcast=false`: non-MoE quantize with ne11_flat=96 for down matmul (wrong)
  - SWIGLU `(i/n)*o0` formula bug (fixed with `j0=i` in unary_gated_op_kernel)

### What's been tried
1. **Eval callback**: fires per compute GROUP not per node — barrier gets bundled
2. **Two-pass inference**: pass 1 captures expert IDs, pass 2 uses them → inherently flawed for multiple reasons. Not even worth thinking about.
3. **Scheduler save/restore**: confirmed slot tensors don't change buffers
4. **EXTERNAL flag on all tensors**: prevents scheduler copies but crashes CUDA assert (buft mismatch)
5. **supports_buft fix**: accept any buft for same device → slot_map recognized as CUDA (tech debt but okay for now)
6. **supports_op for DYN_EX_BARRIER**: barrier assigned to CUDA → GPU spin deadlocks cudaMemcpy
7. **raw_buf_write cudaMemcpyAsync on non-blocking stream**: fixed deadlock, CPU writes while GPU spins
8. **dedup_bcast=false**: avoids scatter path but down matmul quantize gets wrong ne11_flat
9. **all-experts fill**: VRAM unchanged (8 slots) but startup too slow (10K .bin reads)

### Architecture decisions
- **Barrier on CPU**: main thread blocks, CUDA stream free for cudaMemcpy
- **EXTERNAL flag**: scheduler doesn't copy slot tensors/slot_map/barriers
- **cudaHostAllocMapped barriers**: shared CPU/GPU memory, no buffer mismatch
- **Initial fill**: n_slots experts only (fast startup, barrier handles remainder)

### Next
- Fix gate matmul crash: `ids_dst` corruption or scatter-inverse map issue probably. Needs more log debugging.
- Check if up, down and gate sizes match the expected since they arent the same size in q4_k_m

Verify that `strstr(name, "_exps.")` ONLY matches MoE expert tensors and does NOT accidentally skip:
- `token_embd.weight` (embeddings)
- `blk.N.attn_q.weight` (attention Q)
- `blk.N.attn_k.weight` (attention K)  
- `blk.N.attn_v.weight` (attention V)
- `blk.N.attn_output.weight` (attention output)
- `blk.N.ffn_norm.weight` (layer norms)
- `output.weight` (LM head)
- Any tensor with `_exps_` or `_exps.` as a false positive

**Debug approach**: add logging to `load_all_data()` to print EVERY tensor name that is skipped. Verify the skip list contains ONLY expert tensors.

**Note:** barrier is tech debt because it wastes cycles. Good enough for now.

---

## Lessons Learned

### Always use logging
When debugging, add fprintf(stderr, ...) at every step. Never guess the flow. Last session wasted hours on:
- Believing scheduler reallocated buffers (it doesn't)
- Believing callback fired for barrier nodes (it doesn't reach splits)
- Believing two-pass could work (It doesn't because it wastes too much compute and is inherently a flawed idea)
- Believing the barrier thread vs graph buffer mismatch was the issue (it was, for the thread reading cache tensors)

### Always commit
Commit small, commit often. Every working piece of infrastructure should be committed before moving to the next.
Dont' be careless with 'git checkout'. There were too many times when the checkout reverted important stuff when the fix was a single line.

### The predictor is best-effort, not source of truth
The graph's `ggml_argsort_top_k` is the actual expert selection. The MLP predictor is an optimization for prefetching — it guesses, the graph decides. Never use predictor output as `selected_experts_in`.

### Two-pass doesn't work
We cannot say for sure what experts the next layer will need and running the same layer twice wastes insane amounts of compute.

### Ask the user
The user is here to be your critical thinking and workmate. You are not an assistand workhorse but a workmate. Co-operate. Not single player. If there is a decision that comes up then ask the user for their input. Always explain to the user properly don't half ass it.