# Dyn-Ex Optimizations

## Completed (in `de05a8e5d`)

### 1. LIFO Freelist for Group Claiming
**Files**: `llama-dyn-ex.h`, `llama-model.cpp`, `llama-context.cpp`

Replaces atomic CAS loop (scans all 8 groups) with a `std::vector<int> free_groups`. Claim = `pop_back()` (O(1)), release = `push_back()` (O(1)). Eliminates 640 atomic ops and array scans per token (40 layers × 8 CAS checks + 8 LRU scans).

**Speed**: ~10-15µs per layer saved, ~0.5ms per token.

### 2. Persistent Buffers
**Files**: `llama-dyn-ex.h`, `llama-dyn-ex.cpp`, `llama-context.cpp`

Pre-allocates `cpu_buf` (max expert param size, ~860KB), `ids_buf`, `slots_buf` at init time. Eliminates per-callback heap allocation/deallocation (40 `vector::resize` + 40 `new[]`/`delete[]` pairs per token).

**Speed**: ~50-100µs per layer for the 600KB buffer allocation, ~2-4ms per token.

### 3. n_ubatch Doubled
**File**: `llama-dyn-ex.cpp`

`n_ubatch = n_l1 / n_expert_used` instead of `(n_l1 / n_expert_used) / 2`. Allows more tokens per ubatch batch. For L1=64: 8-token ubatches instead of 4. Fewer scheduler invocations and graph rebuilds during prefill.

**Speed**: Fewer graph compute calls during prefill.

### 4. CPU-Forced Virtualized Tensors
**File**: `llama-model-loader.cpp`

Expert tensors (ne[2]=64) allocated on CPU buffer type, not GPU. Scheduler doesn't put them in GPU pool — they're runtime-filled from `.bin`. Keeps GPU memory free for compute tensors.

**Speed**: Reduces GPU memory pressure, avoids OOM during init.

### 5. t_selected_experts Tracking
**Files**: `llama-graph.h`, `llama-graph.cpp`

Per-layer pointer array stores `selected_experts` graph tensor pointer during graph build. Breaks the barrier→argsort dependency edge, allowing ggml's topk-moe fusion optimization to work. Without this, argsort and topk can't be fused because the barrier consumes the argsort output.

**Speed**: Enables GPU kernel fusion, reducing kernel launch overhead.

### 6. CUDA Kernel for Slot Assignment
**Files**: `ggml-cuda.cu`, `llama-dyn-ex.h`

`dyn_ex_slot_assign_kernel` runs on GPU compute stream. Takes `selected_experts` + L2 state, writes `slot_ids` + `miss_flags`. Eliminates CPU computation and CPU→GPU copy of slot_ids. Kernel writes directly to GPU tensor that MUL_MAT_ID reads.

**Speed**: One D2H + one H2D saved per barrier (64 bytes × 40 layers = 2.5KB). Future: enables GPU-side L2 caching.

### 7. CPU Backend DYN_EX_BARRIER No-Op
**File**: `ggml-cpu.c`

Adds `GGML_OP_DYN_EX_BARRIER` as no-op in CPU backend compute/task handlers. Without this, CPU-only or mixed CPU/GPU inference crashes on the unsupported barrier op.

**Speed**: Correctness fix, no speed impact.

### 8. Removed Per-Callback cpu_buf Allocation
**File**: `llama-context.cpp`

Old code: `std::vector<uint8_t> cpu_buf(max_size)` allocated in every callback. New code: uses persistent `de->cpu_buf`. Same as #2 above.

**Speed**: Covered by #2.

### 9. Removed L2 Cache Logic from Callback
**File**: `llama-context.cpp`

Removed ~100 lines of L2 cache lookup (mutex-protected slot_of scan, LRU eviction, L2 background fill). For L2=0 (disabled), this was dead code. GPU kernel handles L2 when enabled.

**Speed**: ~20-50µs per callback for skipped mutex and array scans.

### 10. Removed on_group_release Callback
**File**: `llama-context.cpp`

Removed dead function pointer call (`de->on_group_release`). No consumer registered.

**Speed**: Negligible.

---

## From `122223f11` (To Be Implemented Correctly)

### 11. Post-Based Flow
**File**: `llama-context.cpp`

Pre-callback: only claims group (mutex-guarded LIFO pop), returns `true` (boundary). Post-callback: D2H reads `selected_experts`, loads weights from `.bin`, writes `slot_ids`. Creates explicit GPU sync point between weight H2D and MUL_MAT_ID compute.

Why `de05a8e5d` does pre-based: simpler, no-post-needed. But pre-based relies on implicit same-stream ordering — no explicit sync between `ggml_backend_tensor_set_async` H2D and subsequent MUL_MAT_ID. Post-based makes this explicit and verifiable.

Implementation:
```cpp
// Pre: claim group, return true (boundary)
if (pre) { claim_group(); return true; }
// Post: GPU synced, load weights
read_experts(); load_weights(); write_slot_ids();
return true;
```

### 12. Non-Barrier Nodes Return False (Batched)
**File**: `llama-context.cpp`

`if (t->op != GGML_OP_DYN_EX_BARRIER) return false;` — only barriers create subgraph boundaries. All non-barrier nodes (MUL_MAT_ID, reshape, SiLU, etc.) batched into one subgraph. Eliminates ~3800 subgraph boundaries per token.

Broken in `122223f11` because: without post-based flow, the GPU sync at barriers was missing, and H2D weight copies ran concurrently with MUL_MAT_ID on the same stream with no ordering guarantee. With post-based flow (barrier boundary + post sync), MUL_MAT_ID runs in the NEXT subgraph after the barrier sync, guaranteeing ordering.

Performance: ~3800 fewer subgraph boundaries × 10µs GPU sync = ~38ms saved per token.

### 13. Full L1 Usage During Prefill
**File**: `llama-context.cpp`

`n_slots = min(n_e, n_l1)` — uses all L1 slots, not clamped to `n_expert_used`. Combined with `base_slot` fix (offset = base_slot + i, not i), load all selected experts for the ubatch into the claimed groups' L1 space. For prefill with n_e=32: loads 32 experts across 4 claimed groups.

Requires post-based flow (#11) for correctness — prefill needs boundary separation between layers to prevent L1 overwrite.

### 14. GPU Kernel Hot Path
**File**: `ggml-cuda.cu`

The `dyn_ex_slot_assign_kernel` exists but is dormant (commented out as "TEMP: CPU loading"). Wire it up: H2D expert IDs → kernel launch on compute stream → kernel writes `slot_ids` + `miss_flags`. Combined with #11 (post-based), kernel runs after GPU sync in post.

---

## Future: GPUDirect Storage

### 15. GDS: GPU Reads Expert Weights Directly from NVMe
**Files**: New CUDA code

GPU issues DMA reads from NVMe `.bin` file directly to L1 tensor memory via `cuFileRead`. CPU only queues the request (~5µs) and returns. DMA completes on PCIe bus; MUL_MAT_ID reads L1 on same bus — naturally ordered.

**Requirements**: NVMe on same PCIe switch as GPU (met), `.bin` page-aligned (met), Pascal or newer GPU (GTX 1050 Ti = Pascal, met), NVIDIA driver with GDS support.

**Speed**: Per-expert cost drops from ~150µs (pread + H2D) to ~5µs (queue DMA). ~30x speedup.

| Operation | Current | GDS |
|-----------|---------|-----|
| Per-expert load | 150µs | 5µs |
| Per-token (960 experts) | 144ms | 5ms |
| Max t/s | ~7 | ~200 (PCIe-bound) |

**Combined with Post-Based Flow (#11)**: CPU queues all DMA for a layer in post-callback, returns immediately. GPU processes DMA while computing MUL_MAT_ID for previous layer. Pipelined: compute overlaps with data movement.

**Combined with Pinned L2 (#6)**: GPU kernel checks pinned L2 state. Hit → skip DMA. Miss → queue DMA. L2 fills between tokens from CPU, non-blocking during inference.

---

## Performance Summary

| Optimization | Speed Gain | Status |
|-------------|------------|--------|
| Persistent buffers (#2) | ~3ms/token | **Done** |
| LIFO freelist (#1) | ~0.5ms/token | **Done** |
| Removed L2 logic (#9) | ~1ms/token | **Done** |
| n_ubatch doubled (#3) | Prefill only | **Done** |
| Non-barrier batching (#12) | ~38ms/token | **TODO** |
| Post-based flow (#11) | Correctness | **TODO** |
| Full L1 prefill (#13) | Correctness | **TODO** |
| GPU kernel hot path (#14) | Minor | **TODO** |
| GDS (#15) | ~30x | **Future** |
