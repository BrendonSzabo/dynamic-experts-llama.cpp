# Flash-MoE Architecture (9dafe78 - working)

## Model Loading
1. `llama_model_loader::create_tensor()` takes optional `dim_override` parameter
2. `flash_moe_slot_bank_virtualization_active` flag set during init
3. Two key hooks in create_tensor:
   a. **dim_override path** (line 1274): if dim_override present, create `ovr` tensor with ALL dimensions from override, recompute ALL nb from scratch (nb[i] = nb[i-1] * ne[i-1]), use `ggml_dup_tensor(ctx, &ovr)` 
   b. **CPU forcing** (line 1227): if slot_bank active AND `flash_moe_is_routed_tensor_name()`, force `buft = ggml_backend_cpu_buffer_type()`

4. `llama_model_base::create_tensor()` (model-level wrapper): when flash-moe slot-bank is active and tensor matches routed names, passes `dim_override = {ne0, ne1, n_slots}` to loader's create_tensor

5. Result: routed expert tensors created with ne[2]=n_slots, ALL nb correct, on CPU

## Expert Data Loading
1. Sidecar files: per-layer `.bin` files + `manifest.json`
2. `llama_flash_moe_slot_runtime::install_loads()` reads from sidecar into L2 (CPU) slot banks
3. L1 (GPU) slot banks: Q8_0 tensors, `ggml_backend_alloc_ctx_tensors_from_buft` allocated
4. Data flow: sidecar → L2 CPU → (optional) L1 GPU Q8_0 promotion
5. For decode without CUDA graphs: scheduler copies L2 CPU → GPU compute buffer at graph start

## Graph Building (build_moe_ffn)
1. Gate → topk → selected_experts (expert IDs)
2. **Slot ID injection**: `slot_runtime->build_slot_ids_tensor(ctx0, selected_experts, il)` creates `ggml_map_custom1` op that translates expert IDs → slot IDs at runtime
3. `selected_experts_mm` = slot IDs (not expert IDs)
4. MUL_MAT_ID uses `selected_experts_mm` with CPU-virtualized weight tensors
5. Weight extraction (get_rows) uses original `selected_experts` (expert IDs)
6. **Prefill short-circuit**: `build_prefill_moe_tensor` returns L1 GPU tensors directly

## Runtime (per-token)
1. **Eval callback**: `slot_runtime->handle_tensor(ffn_moe_topk)` fires when gate output computed
2. Reads expert IDs from gate output
3. LRU slot assignment in L2 CPU bank
4. `install_loads()` writes expert data from sidecar into L2 CPU slots
5. For next token: scheduler copies updated L2 → GPU compute buffer
6. First token uses GGUF-loaded experts (0 through n_slots-1)

## Key Architectural Properties
- Virtualized tensors on CPU → scheduler doesn't see them in GPU pool
- GPU compute buffer allocation identical to stock (no ne[2] impact on GPU memory)
- No fused kernel issues (older upstream, fused kernels not as aggressive)
- Slot ID translation via graph op (not callback in-place modification)
- L1 GPU bank optional (Q8_0 promotion for CUDA graph capture)
