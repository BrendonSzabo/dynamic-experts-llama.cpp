# Flash-MoE vs Dynamic Experts: Key Differences

## 1. Tensor Creation
| | Flash-MoE | OURS |
|---|---|---|
| Method | `dim_override` parameter passed to create_tensor | `dyn_ex_n_slots` loader flag |
| nb computation | Fresh ovr, ALL nb from scratch | Fresh ovr, ALL nb from scratch (same) |
| CPU force | `ggml_backend_cpu_buffer_type()` via `flash_moe_slot_bank_virtualization_active` | `ggml_backend_cpu_buffer_type()` via dyn_ex_n_slots (same!) |

## 2. Slot ID Translation
| | Flash-MoE | OURS |
|---|---|---|
| Method | `ggml_map_custom1` graph op (build_slot_ids_tensor) | Barrier callback writes in-place |
| Tensor modified | `selected_experts_mm` (separate tensor) | `selected_experts` (same tensor, in-place) |
| When | During graph compute (custom op runs on GPU) | Before graph compute (eval callback) |

## 3. Expert Data Loading
| | Flash-MoE | OURS |
|---|---|---|
| Source | Sidecar files (per-layer .bin + manifest.json) | Single .bin file (VLLM\x02 format) |
| Reader | `ExpertFileReader` | `dyn_ex_reader` (O(1) mmap) |
| Write target | L2 CPU slot bank → (optional) L1 GPU Q8_0 | Model tensor data directly (memcpy) |

## 4. Graph Structure
| | Flash-MoE | OURS |
|---|---|---|
| Barrier | No barrier op | GGML_OP_DYN_EX_BARRIER |
| Slot injection | `build_slot_ids_tensor` (ggml_map_custom1) or graph input | In-place modification of selected_experts |
| Prefill path | `build_prefill_moe_tensor` (L1 GPU tensors) | None |

## 5. Runtime Order
| | Flash-MoE | OURS |
|---|---|---|
| 1 | Scheduler copies CPU→GPU | Scheduler copies CPU→GPU |
| 2 | Graph compute: gate → topk | Graph compute: gate → topk |
| 3 | ffn_moe_topk computed | BARRIER fires |
| 4 | **Eval callback fires**: reads IDs, loads to L2 CPU | **Eval callback fires**: reads IDs, loads to model tensors (memcpy) |
| 5 | Next token: scheduler copies L2→GPU | MUL_MAT_ID runs with stale GPU copies (copy happened before barrier) |

## THE BUG
Line 5 is the timing problem. Both Flash-MoE and OURS have the scheduler copy BEFORE the barrier. The barrier writes new data AFTER the copy. MUL_MAT_ID reads the STALE COPY from before the barrier.

**Flash-MoE works because**: the first token uses GGUF defaults (experts 0-n_slots). The barrier loads correct experts into L2 CPU. On the NEXT token, the scheduler copies the updated L2 data to GPU. Each token uses the previous token's expert selection (one token delayed).

**OURS crashes because**: the first token also uses GGUF defaults. The barrier loads correct experts. But the **fused RMS norm kernel** crashes because it reads a tensor that the scheduler placed on CPU. The fused kernel tries to access the CPU tensor directly from the GPU.

## THE ACTUAL FIX
The fused kernel reads a tensor that was on CPU. Which tensor? It could be:
- A virtualized expert tensor (MUL_MAT_ID weight) that wasn't copied to GPU
- An intermediate tensor derived from the virtualized tensor
- A norm weight that was accidentally placed on CPU

The `rms_norm_mul_f32_cuda` crash at frame 7 shows the kernel reads `src0`, `src1`, and other tensors. If ANY of these are on CPU, the GPU kernel crashes.

**Solution**: The scheduler must copy ALL CPU tensors needed by the fused kernel BEFORE graph compute. Currently, the scheduler only copies tensors for MUL_MAT/MUL_MAT_ID. Fused ops read tensors that may not be in the copy list.
