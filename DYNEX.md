# Dynamic Experts Architecture (dyn-ex-v4)

## Model Loading
1. `llama_model_loader` has `dyn_ex_n_slots` flag set during init
2. In `create_tensor()`, two hooks:
   a. **First hack** (line 1252): if `t_meta->op == GGML_OP_MUL_MAT_ID`, modify `t_meta` in-place: set `ne[2]=n_slots`, recompute `nb[0..2]`, but `nb[3]=nb[2]` (BUG: should be `nb[2]*ne[2]`). Adjusts `size_data`.
   b. **Second hack** (line 1279): if `ne[2] > n_slots`, create fresh `ovr` tensor with override `ne[2]=n_slots`, recompute ALL `nb` correctly via `nb[i]=nb[i-1]*ne[i-1]`, use `buft_for_tensor(&ovr)` (GPU). **Force CPU** (line 1299): `buft = ggml_backend_cpu_buffer_type()`.

3. Result: routed expert tensors created with ne[2]=n_slots on CPU. First hack modifies GGUF tensor in-place (bad), second hack overrides via ovr (good).

## Expert Data Loading
1. Single `.bin` file (VLLM\x02 format) with per-expert Q4_K/Q6_K blocks
2. `dyn_ex_reader_open()` mmaps the file, O(1) lookup via `param_data_off + (layer*n_experts+expert)*stride`
3. `dyn_ex_cache_init()` reads .bin metadata (param types, shapes, sizes)
4. `dyn_ex_cache_alloc_barriers()` creates barrier tensors (EXTERNAL, no buffer)

## Graph Building (build_moe_ffn)
1. Gate → topk → selected_experts (expert IDs)
2. **Barrier node**: `dyn_ex_barrier[il]` added to graph with `bar->src[0] = selected_experts`
3. `selected_experts_slots = selected_experts` (same tensor, in-place modification by callback)
4. MUL_MAT_ID uses `selected_experts_slots` 
5. **No slot ID translation graph op** — callback modifies tensor in-place

## Runtime (per-token)
1. **Eval callback** fires on barrier node (GGML_OP_DYN_EX_BARRIER)
2. Reads expert IDs from `src->data` (selected_experts on GPU)
3. Loads experts from .bin into CPU buffer
4. Writes to model tensors via `memcpy` (CPU tensors)
5. Writes slot IDs (0,1,2,... sequential) back to `src->data` via `cudaMemcpy`
6. First token uses GGUF-loaded experts (0 through n_slots-1)

## Key Architectural Differences from Flash-MoE
- **No slot ID translation op** — callback modifies tensor in-place
- **memcpy writes to CPU tensors** — needs scheduler copy for GPU access
- **Fused kernel crash**: `rms_norm_mul_f32_cuda` reads CPU tensor in chain
- **First hack modifies GGUF tensor in-place** — potential nb[3] bug
- **No L1 GPU slot bank** — tensors are directly on CPU/GPU
