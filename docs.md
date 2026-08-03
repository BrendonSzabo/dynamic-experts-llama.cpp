# Dynamic Experts

## Setup
- Model: Qwable-3.6-35b (Qwen35MoE arch), Q4_K_M, 40 layers, 256 experts, n_expert_used=8
- GPU: NVIDIA GTX 1050 Ti, 4GB VRAM, compute 6.1
- VRAM: 1884 MB model + 5 MB KV + 63 MB recurrent + 9 MB compute ≈ 1960 MB

## Root Cause: Q4_K_M Mixed Quant Types

The model is Q4_K_M — a mix of Q4_K (type 12) and Q6_K (type 14) tensors per layer. 20 layers have Q4_K down_exps (589 KB/expert), 20 have Q6_K down_exps (860 KB/expert). Gate and up tensors are all Q4_K.

The `.bin` file stores ONE dtype per param (from layer 0). The callback was using `.bin`-derived per-expert sizes, which matched the Q6_K layers but overflowed Q4_K layer tensors by 270 KB per slot → GPU buffer overflow → garbage output.

**Fix**: Use `tensor->nb[2]` (the actual tensor stride) for per-expert sizing at runtime. Each layer's tensor knows its own type. Created dual L1 tensors for `down_exps` (one Q4_K, one Q6_K), assigned per-layer based on GGUF type.

## Architecture: 3-Level Cache

```
L1 (GPU, global): per-layer ext_slot tensors with ne[2]=n_l1
                  --dyn-ex-l1 N  (e.g., 8)
                  
L2 (Host, per-layer): per-layer host buffers, LRU eviction
                      --dyn-ex-l2 N  (e.g., 64)
                      O(1) lookup via slot_of[expert_id]

L3 (.bin mmap): existing reader
```

**Load path**: L3 → L2 → L1 (if L2 enabled), or L3 → L1 direct (if L2=0).

**Cache details**:
- L2: per-layer, `n_l2` slots × `per_expert_bytes`, LRU eviction via `age[]` array + `slot_of[expert]` for O(1) lookup
- Mutex-protected eviction (for future parallel loading)
- Background L2 fill: GPU copies happen first, then L2 is populated from `.bin` for future reuse
- `MADV_WILLNEED` on the .bin mmap for warm page cache

## Callback: 3-Phase Loading

1. **Phase 1** (mutex): Resolve L2 slots, O(1) lookup, LRU eviction
2. **Phase 2**: Load expert data from L2 (hit) or `.bin` (miss) into staging buffer, then batched `cudaMemcpy` to GPU (4 calls per layer instead of 96)
3. **Phase 3**: Background L2 fill from staging (zero-copy for misses)

## Managed Memory for Slot IDs

Slot IDs were written via `cudaMemcpy` (H→D) in the callback. Replaced with `cudaHostAllocMapped` — CPU writes directly to managed memory, GPU reads via MUL_MAT_ID. Zero-copy. Eliminates one `cudaMemcpy` per layer.

The `build_moe_ffn` graph builder no longer creates slot_ids tensors — uses pre-allocated managed tensor stored in `bar->src[1]`, set during `dyn_ex_cache_init_managed`.

CUDA events for GPU/CPU overlap were attempted but ran into stream issues. Managed memory alone eliminates the copy.

## Profiling

Per-token timing (logged every 40th layer, i.e., per token):

```
dyn-ex: read=0.0 p1=0.1 bin=5.3 dma=2.0 l2fill=0.8 ms
dyn-ex DEC #N: compute=250 ms
```

| Phase | Time | What |
|---|---|---|
| read_ids | 0.0ms | GPU→CPU copy of 8 ints |
| p1 | 0.1ms | mutex + O(1) L2 lookup |
| bin | 0-20ms | `.bin` mmap reads (misses) |
| dma | 1-3ms | cudaMemcpy H→D (32 small calls) |
| l2fill | 0-2ms | Background L2 fill |

Expert DMA: ~15.6 MB/token from the callback. CUDA H→D counter (in `ggml_backend_cuda_set_tensor_async`) confirms no unexpected scheduler copies.

## Things Tried

- **Staging buffer batching**: Reduced GPU copies from 96 to 4 per layer. Saves ~1ms vs 32 individual calls.
- **Direct DMA from L2**: Skipping staging — hit time dropped from 7ms to 5.5ms, but miss time similar.
- **Async GPU copies**: `cudaMemcpyAsync` + dedicated stream — added overhead, removed in favor of batched sync copies.
- **GPU events + barrier**: Attempted `cudaStreamWaitEvent` to overlap CPU/GPU but hit stream/sync issues. Managed memory alone is sufficient.
- **MADV_WILLNEED**: Pre-faults `.bin` pages. First-token bin time dropped from 27ms to ~10ms.
- **CUDA graphs**: Currently disabled for dyn-ex — graph replay was 400-1500ms vs 200ms without graphs. Re-enabled after managed memory fix for testing.

## Current Performance

| Metric | Value |
|---|---|
| Per-token compute | ~250ms avg |
| Throughput | 4-5 t/s |
| Expert DMA | 15.6 MB/token |
| Callback overhead | ~5-7ms/token (2-3% of total) |

GPU compute (MUL_MAT_ID × 40 layers, Q4_K/Q6_K GEMV) dominates at ~240ms/token. The callback and expert loading are negligible overhead.

## Trace Capture (`--trace DIR`)

Saves per-token tensor states. Works without dyn-ex (CPU or GPU). Call `llama_trace_flush()` after generation to write `DIR/trace.bin`.

Captured tensors per token:
- `inp_embd` — token embeddings (model input)
- `ffn_inp-{layer}` — hidden state after attention, before MoE (per layer)
- `ffn_moe_out-{layer}` or `ffn_out-{layer}` — hidden state after MoE (per layer)
- `result_norm` — final RMS norm output
- `result_output` — logits (pre-softmax, vocab-sized)

### Binary Format

Single file `DIR/trace.bin`, all tokens concatenated. Each entry:

```
entry:
  token_id  u32      token index (0, 1, 2, ...)
  name_len  u32      length of tensor name in bytes
  name      char[]   tensor name (not null-terminated)
  ne[0]     u32      dimension 0 (elements)
  ne[1]     u32      dimension 1
  ne[2]     u32      dimension 2
  ne[3]     u32      dimension 3
  nbytes    u32      raw data size in bytes
  data      u8[]     raw tensor data (row-major, dtype = tensor type)
```

Entries are buffered in memory during inference, written on `llama_trace_flush()`.

### Reading with Python

```python
import struct

def read_trace(path):
    entries = []
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: break
            token_id, name_len = struct.unpack('<II', hdr)
            name = f.read(name_len).decode()
            ne = struct.unpack('<IIII', f.read(16))
            nbytes = struct.unpack('<I', f.read(4))[0]
            data = f.read(nbytes)
            entries.append({'token': token_id, 'name': name, 'ne': ne, 'data': data})
    return entries
```
