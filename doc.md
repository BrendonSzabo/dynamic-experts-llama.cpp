# Dynamic Experts v2 — Architecture Redesign

## 0. Current Status (2026-08-01)

### Working
- .bin reader: O(1) lookup, verified byte-perfect against GGUF
- .bin data: MATCHES GGUF exactly — Python cross-reference confirmed E213, E238, E193 identical
- GPU slot data: verified byte-for-byte against .bin for all 8 selected experts per layer
- Slot_map: verified correct on GPU, remap produces correct [0..7]
- Strides: nb values match original GGUF tensors
- Stock llama.cpp kernels handle ne02=16 (MUL_MAT_ID dispatches to vec_q)
- Barrier callback: loads experts before matmul, operates correctly
- suports_buft: stock code already accepts our EXTERNAL buffers (buft->device == dev)
- Model runs end-to-end, generates 76 tokens at 3.7 tok/s on Pascal GTX 1050 Ti

### Failing
- vec_q down matmul with ne11=8 (multi-channel SWIGLU input) produces NaN/wrong values
  - Gate matmul ne11=1 (broadcast): works correctly
  - Down matmul ne11=8: vec_q quantize stride bug for ne02=16
  - This is a stock llama.cpp vec_q kernel bug triggered by our slot tensor dimensions

### Root Cause: vec_q Early Return Bug (FOUND & FIXED)
- `mmvq.cu` had a leftover debug return: `if (ids && src0->ne[2]==16) return;`
- This skipped ALL gate/up/down matmuls for slot tensors
- Removed — gate matmul now works correctly

### Remaining: vec_q Multi-Channel Bug
- Down matmul (`ffn_moe_down`): vec_q with ne11=8, ne02=16
- Quantized activation layout differs from vec_q's expected layout
- Produces massive wrong values (-8.6M, -28M) or NaN
- Workaround: route down matmul through sorted path (individual experts)
  - Or: increase n_slots to push dispatch to MMQ path

## 0.1 What We Learned

1. **The ggml buffer approach works**. `supports_buft` already accepts buffers from the same device.
   No cudaMalloc bypass needed — the architecture is correct as-is.

2. **Stock llama.cpp kernels handle Q4_K + ne02=16 correctly** for single-channel (ne11=1).
   The multi-channel (ne11=8) path has a vec_q quantize stride bug.

3. **The .bin data is correct**. The interleaved-blocks theory was wrong — the .bin already
   has per-expert data laid out correctly. Byte-for-byte verification confirmed this.

4. **git checkout is dangerous**. Multiple reverts destroyed work. Always commit+push before checkout.

5. **Debug returns are insidious**. The vec_q early-return was a "temporary" debug hack that
   stayed in the code for days, causing us to chase phantom bugs.

## 1. Architecture (Correct)

### 1.1 Memory Management

The current ggml buffer approach works:
- Slot buffers via `ggml_backend_buft_alloc_buffer` (same device, accepted by supports_buft)
- EXTERNAL flag on slot tensors prevents scheduler reallocation
- `raw_buf_write` for GPU writes (cudaMemcpyAsync on non-blocking stream)
- No need to switch to cudaMalloc — stock ggml handles it

### 1.2 Kernel Paths

- Gate matmul (ne11=1): vec_q → correct
- Up matmul (ne11=1): vec_q → correct  
- Down matmul (ne11=8): NEEDS SORTED PATH bypass
  - Option A: `ne11 > 1` check in vec_q eligibility → falls to sorted
  - Option B: Larger n_slots → dispatches to MMQ

### 1.3 Verified Data Pipeline

```
GGUF (Q4_K interleaved) ──convert script──▶ .bin (Q4_K per-expert)
                                                  │
                                          dyn_ex_reader (O(1))
                                                  │
                                          CPU buffer (read)
                                                  │
                                          raw_buf_write → GPU slot
                                                  │
                                          get_experts / ensure_ordered
                                                  │
                                          MUL_MAT_ID (ne02=16)
```

All stages verified byte-for-byte.

## 2. What Failed (and Why)

### 2.1 vec_q Early Return (FIXED)
- `mmvq.cu:1167`: `return` without computing for ne02==16
- Debug hack from vec_q debugging era, never removed
- Skipped ALL gate/up/down matmuls for slot tensors
- FIX: removed

### 2.2 should_use_mmq Bypass (REMOVED)
- `mmq.cu:304`: `n_experts < 128 → return false`
- Prevented MMQ from handling slot tensors on Turing+
- REMOVED

### 2.3 nan Check (REMOVED)
- `mmvq.cu:1260`: scanned 512 output elements for NaN
- Diagnostic code, harmless but clutter
- REMOVED

### 2.4 Kernel Bypasses (All Removed)
- vec_q/MMQ bypass via early returns — wrong because callers always `return` after calling
- Fusion bypass via `src0->ne[2] >= 128` — unnecessary
- Sorted path was never reached because of #2.1

### 2.5 FP16 Dequantize Experiment (Incomplete)
- GPU dequantize kernel written (`llama-dyn-ex-dequant.cuh`)
- Blocked by ggml include path issues
- CPU dequantize tried but too invasive
- Reverted

### 2.6 ne02=256 Experiment (Failed)
- Changed ne[2] to 256 to force original kernel dispatch
- `ggml_nbytes` mismatch caused assertion failure
- Reverted

## 3. Next Steps

### Immediate Fix
Add `ne11 <= 1` guard in vec_q eligibility for MUL_MAT_ID:
```cpp
if (ne2 <= MMVQ_MAX_BATCH_SIZE && ne11 <= 1) {
    // vec_q for gate/up matmuls (ne11=1, works)
    // Down matmul (ne11=8) falls through to sorted path
}
```

### Quality Verification
1. Apply vec_q bypass for ne11>1 only
2. Test with stock llama.cpp CPU for first-token comparison
3. If output matches: architecture validated
4. If not: investigate vec_q gate matmul numerical differences

### Performance
1. Measure sorted path overhead (individual expert matmuls)
2. If too slow: implement FP16 dequantize cache for down matmul
3. Add CUDA stream for async copies during prefetch

## 4. Cleanup

### Completed
- All kernel bypasses removed
- should_use_mmq restored to stock
- vec_q early return removed
- Per-op CPU+CUDA logging added for comparison

### Remaining
- vec_q multi-channel bypass (ne11>1 guard)
- Remove unused FP16 buffers from cache struct
- Clean up debug logs
