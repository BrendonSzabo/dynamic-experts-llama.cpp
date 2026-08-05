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
