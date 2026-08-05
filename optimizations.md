# Dyn-Ex Optimizations

## Architecture

### Three-Tier Storage
```
L1: GPU L1 tensor     → ne[2] = n_l1 slots, split into n_groups = n_l1/n_expert_used groups
L2: Pinned host memory → per-layer LRU cache, GPU-readable via device pointer
L3: .bin file (mmap)   → on-disk expert weights, page-aligned
```

- **L1**: One group = n_expert_used slots = enough for one layer during single-token decode.
- **L2**: `cudaHostAlloc(cudaHostAllocMapped)` pinned memory. GPU reads via device pointer — no CPU. CPU writes between tokens via host pointer (zero-copy). Not live-filled during inference.
- **L3**: mmap'd `.bin`, page-aligned. Loaded via `pread` (CPU) or GDS `cuFileRead` (GPU DMA).

### Prefill vs Decode
| | Prefill | Decode |
|---|---|---|
| n_ubatch | 1 | 1 |
| n_tokens/ubatch | max(1, n_l1/n_expert_used) | 1 |
| L1 usage | Full L1 — all n_l1 slots, no grouping | One group per layer via FIFO |
| Claim boundary | Yes (sequential layers) | No (contiguous graph) |
| Expert count | n_e ≤ n_l1 | n_e = n_expert_used |

**Graph distinction**: Prefill and decode produce different graphs (different `n_tokens` → different tensor dimensions). The scheduler cannot reuse one for the other — each gets built fresh. The callback detects the case via `n_e = src->ne[0] * src->ne[1]`: `n_e == n_expert_used` is decode (return false, contiguous), `n_e > n_expert_used` is prefill (return true, sequential). No flag needed — the graph shape IS the flag.

### Expert Loading
```
for each layer's selected experts:
    L1 ← try L2 hit: yes → load to L1 slot, no → L3 (.bin) → L1, record miss
```
- **During token**: L2 is read-only. Kernel reads L2, loads from L3 on miss.
- **Between tokens**: CPU reads miss table from pinned memory, fills L2 from L3, GPU sync.

### Data Movement
- **Expert loading**: Pure CUDA. GDS: `cuFileRead` NVMe→L1 DMA. Fallback: CPU pread → pinned → cudaMemcpyAsync H2D.
- **L2 state**: Pinned mapped, device+host pointers. GPU kernel reads/writes device ptr. CPU reads host ptr zero-copy.
- **Slot IDs**: GPU kernel writes to GPU tensor, MUL_MAT_ID reads same tensor. No copy.
- **Graph**: Fully contiguous GPU graph. Barriers return false (batched), non-barriers return false (batched). One subgraph per split. GPU stream orders everything naturally.

---

## Optimization List

### Layer 1 — Data Structure

#### 1. FIFO Freelist for Group Claiming
**Files**: `llama-dyn-ex.h`, `llama-model.cpp`, `llama-context.cpp`

`std::deque<int> free_groups`. Claim = `pop_front()`, release = `push_back()`. O(1), no lock needed — push/pop touch different ends. FIFO distributes groups evenly across layers (cycling through all 8 before reuse). Replaces atomic CAS scan (O(n_groups)).

**Speed**: ~0.5ms/token.

#### 2. Persistent Buffers
**Files**: `llama-dyn-ex.h`, `llama-dyn-ex.cpp`

Pre-allocates `cpu_buf` (~860KB), `ids_buf`, `slots_buf` at init. Eliminates per-callback heap allocation. CPU path only (GDS bypasses these for GPU loading).

**Speed**: ~3ms/token.

#### 3. CPU-Forced Virtualized Tensors
**File**: `llama-model-loader.cpp`

Expert tensors (ne[2]=n_l1) on CPU buffer type. Scheduler doesn't allocate GPU memory — filled at runtime. Keeps GPU memory for compute.

### Layer 2 — Graph Structure

#### 4. Contiguous GPU Graph (No Splits)
**Files**: `llama-context.cpp`, `ggml-cuda.cu`

Barriers return `false` (batched), non-barriers return `false` (batched). One subgraph per GPU split. GPU stream orders H2D → kernel → MUL_MAT_ID. No scheduler sync between layers. Requires pure CUDA loading (no CPU blocking).

**Speed**: Eliminates ~3800 subgraph boundaries. ~38ms/token.

#### 5. Prefill Sequential Claim
Claim returns `true` (boundary) during prefill. Each layer's MUL_MAT_ID runs before next claim. Prevents L1 overwrite when n_e exceeds group size.

### Layer 3 — Pure CUDA Loading

#### 6. GPU Kernel for Slot Assignment
**Files**: `ggml-cuda.cu`, `llama-dyn-ex.h`

`dyn_ex_slot_assign_kernel` on GPU compute stream. Reads `selected_experts` (GPU), L2 (pinned device ptr). Writes `slot_ids` (GPU tensor), `miss_flags` (pinned). Eliminates CPU D2H + H2D.

#### 7. Pinned Mapped L2 State
**Files**: `llama-dyn-ex.h`, `llama-dyn-ex.cpp`

L2 arrays via `cudaHostAlloc(cudaHostAllocMapped)`. Device ptr for kernel, host ptr for CPU. Kernel reads L2 during token (read-only). CPU fills L2 between tokens from miss table. Zero-copy reads.

#### 8. GPUDirect Storage
**Files**: New CUDA code

`cuFileRead` DMA from NVMe `.bin` → L1 tensor. GPU reads expert weights directly, no CPU. NVMe on same PCIe switch. Pascal GPU supports it. Page-aligned `.bin`.

**Speed**: Per-expert: 5µs (queue DMA) vs 150µs (pread+H2D). Per-token: 5ms vs 144ms. ~30x.

---

## Trace Gathering (`--trace <dir>`)

Gathers per-token inference telemetry: hidden states, attention, logits, top-k experts, slot assignments, output token. Uses GPU-side linked list — kernels atomically append during inference, CPU drains after token.

### Design
- **GPU**: Device buffer as linked list. Kernels atomically swap head pointer, write entry. Zero CPU involvement during inference — pure device-side CAS.
- **CPU drain**: After `cudaDeviceSynchronize` at token end, walk list via `cudaMemcpy`, deserialize, write to binary log file in trace directory.
- **Guard**: `#ifdef GGML_USE_CUDA` + runtime `--trace <dir>`. Zero overhead when disabled.

---

## Implementation Order

### Phase 1 — Foundation (no correctness risk, measured speed gains)
| # | Optimization | Files | Scope |
|---|---|---|---|
| 1 | FIFO Freelist | `dyn-ex.h` (+`<deque>`, +field), `model.cpp` (+`push_back` in init), `context.cpp` (claim=`pop_front`, release=`push_back`) | Replace CAS loop with O(1) deque ops. No lock. Expect ~0.5ms/token gain. |
| 2 | Persistent Buffers | `dyn-ex.h` (+fields), `dyn-ex.cpp` (+`resize` in init) | Pre-allocate `cpu_buf`, `ids_buf`, `slots_buf`. Remove per-callback vector allocs. Expect ~3ms/token gain. |
| 3 | Virtualized Tensors | `model-loader.cpp` (1 line: `buft = ggml_backend_cpu_buffer_type()`) | Force expert tensors to CPU. Memory optimization, no speed change. |
| 4 | n_ubatch Doubled | `dyn-ex.cpp` (remove `/2`) | One-line change. Prefill speedup only. |
| 5 | t_selected_experts | `graph.h` (+field in params+context), `graph.cpp` (+store in build_moe_ffn, +constructor init) | Break barrier→argsort dependency for fusion. Mild graph change, verify output unchanged. |

### Phase 2 — Graph Restructuring (correctness-critical)
| # | Optimization | Files | Scope |
|---|---|---|---|
| 6 | Contiguous Graph | `context.cpp` (barriers return false, non-barriers return false), `ggml-cuda.cu` (barrier on GPU) | Eliminate ~3800 subgraph boundaries. Must pair with pure CUDA loading (#8-9) for correctness. Expect ~38ms/token gain. |
| 7 | Prefill Sequential | `context.cpp` (claim returns true when `n_e > n_expert_used`) | Prefill only — creates per-layer boundaries when multi-token. Decode stays contiguous. Safe: prefill/decode are different graphs. |

### Phase 3 — Pure CUDA Loading
| # | Optimization | Files | Scope |
|---|---|---|---|
| 8 | Pinned L2 | `dyn-ex.h` (+host fields), `dyn-ex.cpp` (`cudaHostAlloc`+memset), `ggml-cuda.cu` (kernel reads pinning device ptr) | GPU-accessible L2. Kernel checks L2 during inference. CPU fills between tokens. ~5KB of pinned memory per layer. |
| 9 | GPU Kernel Hot Path | `context.cpp` (wire kernel launch in callback) | Enable dormant `dyn_ex_slot_assign_kernel`. Kernel writes slot_ids + miss_flags on GPU. Pair with pinned L2 (#8). |
| 10 | GDS | New CUDA file + `context.cpp` (queue `cuFileRead` instead of pread) | 30x expert load speed. Requires NVMe on same PCIe switch, Pascal+ GPU. Fallback to pread+H2D if GDS unavailable. |

### Phase 4 — Observability
| # | Optimization | Files | Scope |
|---|---|---|---|
| 11 | Trace Gathering | New `trace.h`/`trace.cpp` + `context.cpp` (drain after token) | GPU-side linked list, atomic append. D2H drain between tokens. Guarded by `--trace <dir>`. Zero overhead when disabled. |

---

## After the Rest Works

#### 12. Skip Expert Tensor Loading During Startup
**Files**: `llama-model-loader.cpp`

When dyn-ex is active, expert weight tensors (MUL_MAT_ID ops) are loaded from `.bin` at runtime — the GGUF copy is never used. Skip reading them from GGUF during model loading to speed up startup time.

**Mechanism**: If `dyn_ex_n_slots > 0` add `TENSOR_SKIP` to non static tensors. The tensor still exists in the graph (shape needed for build), but no weight bytes are loaded from disk.

**Impact**: Saves ~15GB of disk reads during model loading.
