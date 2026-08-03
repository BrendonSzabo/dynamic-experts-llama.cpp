# Dynamic Experts Investigation

## Setup
- Model: Qwable-3.6-35b (Qwen35MoE arch), Q4_K/Q6_K, 40 layers, 256 experts, n_expert_used=8
- GPU: NVIDIA GTX 1050 Ti, 4GB VRAM, compute 6.1
- .bin file: VLLM\x02 format, 19GB, produced by convert-gguf-to-expert-binary.py
- Flash-moe fork at /home/brad/llama.cpp-flash-moe commit 9dafe78c
- Base commit: 8c760f5a2 (slot-ID routing fix, eval callback fix, graph reuse enabled)

## What Works (Flash-MoE)
- Flash-moe produces correct output on this same GPU
- Expert tensors on CPU, scheduler copies CPU→GPU via MoE selective copy (ggml-backend.cpp:1588)
- Uses same MUL_MAT_ID kernel, same quantization, same dimensions
- One-token delay: experts loaded during token T used at token T+1

## What Doesn't Work (Dyn-Ex)
- Model produces garbage output: `!!!!!!!!!!!!!!!!` for any token count
- Layer 0-5 gates produce valid expert IDs (e.g. [238,112,120,...])
- Layer 6-39 gates produce [0,1,2,3,4,5,6,7] — hidden state corrupted by layer 6

## Attempts and Results

### Attempt 1: Eval callback return semantics
- Changed `return false` → `return true` for non-barrier tensors
- Goal: preserve scheduler fused-op split boundaries
- Result: no effect on output quality (still garbage)
- Status: KEPT (cleaner behavior, no downside)

### Attempt 2: Graph reuse
- Removed `graph_reuse_disable = true`
- Goal: enable graph reuse for decode performance
- Result: no effect on output quality, graphs reused = 17-33
- Status: KEPT

### Attempt 3: Slot-ID routing (flash-moe pattern)
- Created separate `ggml_set_input` slot_ids tensor instead of in-place selected_experts modification
- Goal: avoid corrupting expert IDs needed by get_rows/add_id
- Result: no effect on output quality
- Status: KEPT (correctness improvement for bias lookup)

### Attempt 4: cudaDeviceSynchronize
- Added sync after all expert writes per layer
- Goal: ensure GPU sees callback's cudaMemcpy writes
- Result: no effect on output quality
- Status: REVERTED (expensive, no benefit)

### Attempt 5: ggml_backend_tensor_set for expert writes
- Replaced raw cudaMemcpy with backend-aware tensor_set
- Goal: write through scheduler's buffer management
- Result: GGML_ASSERT(offset + size <= ggml_nbytes(tensor)) failed
  - Tensor was on GPU (ext_slot EXTERNAL), nbytes check failed despite correct dimensions
- Could NOT determine why nbytes check fails for GPU EXTERNAL tensors
- Status: REVERTED

### Attempt 6: External buffer as src[3] for MUL_MAT_ID
- Set src[3] = gate_exps (EXTERNAL GPU tensor) on MUL_MAT_ID nodes
- Modified CUDA kernel to read src[3] instead of src[0] when present
- Goal: bypass any scheduler copy by reading from known buffer
- Result: src[0]->data == src[3]->data (same pointer) — no scheduler copy exists
  - Model still produces garbage
- Status: REVERTED (confirmed no copy issue)

### Attempt 7: Data verification (byte-level dump)
- Dumped .bin expert data and GPU tensor data for all layers/slots
- Compared byte-for-byte
- Result: all dumps match — expert data loads correctly to GPU tensor
- Confirmed: .bin reading works, cudaMemcpy writes correctly

### Attempt 8: MUL_MAT_ID kernel parameter logging
- Logged ne, nb, type, path (MMVQ/MMQ/MMF/FALLBACK), dst values
- For decode (n_tokens=1): hits MMVQ single-token path (not MOE kernel)
- Parameters: ne=[2048 512 8], nb=[144 1152 589824], schan_x=4096, srow_x=8
- All stride calculations correct for Q4_K
- ids=[0,1,2,3,4,5,6,7] — correct slot IDs
- dst values look reasonable (not NaN/Inf)
- Status: informative, no fix

### Attempt 9: Expert ID logging (callback)
- Dumped expert IDs from selected_experts per layer
- Layer 0-5: plausible expert IDs (e.g. [238,112,120,148,254,...])
- Layer 6-39: all [0,1,2,3,4,5,6,7] — gate produces sequential IDs
- Indicates hidden state corruption by layer 6

### Attempt 10: CPU tensor approach (flash-moe style)
- Version A: force CPU in create_tensor + remove ext_slot
  - Model loads but hangs during inference (gallocr reallocation loop ~20 cycles/30s)
- Version B: force CPU in create_tensor + keep ext_slot with CPU buffer
  - Model loads but still hangs
- Version C: GPU in create_tensor + ext_slot with CPU buffer
  - Model loads but still hangs
- Root cause: scheduler tries to allocate GPU compute buffers for CPU tensor copies,
  but with 40 layers × 3 tensors × 4.7MB = 564MB of CPU tensors plus existing ~2GB model,
  the 4GB GPU runs out of memory for compute buffer copies
- Flash-moe works because of different scheduler interaction (possibly smaller
  slot counts, different model, or different tensor layout)
- Status: REVERTED (OOM on this hardware)

### Attempt 11: MUL_MAT_ID FP32 kernel test
- Created small FP32 test: ne0=64, ne1=32, ne2=4
- CPU FP32 vs GPU FP32: 0.000000 error (PASS)
- CPU FP32 vs numpy: matches (PASS)
- Confirms: MUL_MAT_ID kernel works correctly for FP32
- Could NOT test Q4_K due to missing quantization API in public headers
- Status: informative

### Attempt 12: Tensor packing comparison
- Compared convert-gguf-to-expert-binary.py vs flashmoe_sidecar.py
- Dyn-ex: reshapes GGUF bytes → [n_experts, per_expert_bytes] → writes per-expert
- Flash-moe: copies EXACT raw GGUF bytes verbatim per tensor
- Both produce same per-expert byte content (verified by dump)
- Status: no difference found

## Remaining Hypotheses
1. Q4_K dequant produces different values on GPU vs CPU at expert boundaries
2. Stride nb[3] for virtualized tensor differs from what kernel expects
3. Buffer type (WEIGHTS vs COMPUTE) affects how CUDA allocator handles the tensor
4. Some model-specific issue with Qwen35MoE shared experts interacting with virtualization

## Test Files
- `/tmp/mulmat_test/` — FP32 weights, input, ids, numpy expected
- `/tmp/test_gpu_fp32` — GPU vs CPU FP32 MUL_MAT_ID test binary
- `tools/test_mul_mat_id.cpp` — standalone test (needs CMake target)
