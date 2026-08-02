#include "llama-moe-cuda-graph.h"

// ---------------------------------------------------------------------------
// llama_moe_cuda_graph_capture: initialize per-layer flags for callback shortcut.
// No actual CUDA graph is captured — the speedup comes from the eval callback's
// return-false path, which batches FFN tensors into larger sub-graphs (fewer GPU
// sync points). The CUDA backend's built-in USE_CUDA_GRAPH mechanism also captures
// and replays the MoE sub-graph on the 2nd+ token.
// ---------------------------------------------------------------------------

bool llama_moe_cuda_graph_capture(
    llama_moe_cuda_graph_context & ctx,
    size_t                         n_layers)
{
    ctx.n_layers = n_layers;
    ctx.layer_skip_experts.resize(n_layers, false);
    ctx.layer_replay_succeeded.resize(n_layers, false);
    ctx.layer_norm_ptrs.resize(n_layers, nullptr);
    ctx.layer_residual_ptrs.resize(n_layers, nullptr);

    // Mark all layers as needing the callback shortcut
    for (size_t i = 0; i < n_layers; i++) {
        ctx.layer_skip_experts[i] = true;
    }

    ctx.any_captured = true;
    return true;
}

// ---------------------------------------------------------------------------
// llama_moe_cuda_graph_destroy: reset flags.
// ---------------------------------------------------------------------------

void llama_moe_cuda_graph_destroy(
    llama_moe_cuda_graph_context & ctx)
{
    ctx.layer_skip_experts.clear();
    ctx.layer_replay_succeeded.clear();
    ctx.layer_norm_ptrs.clear();
    ctx.layer_residual_ptrs.clear();
    ctx.n_layers = 0;
    ctx.any_captured = false;
}

#ifdef GGML_USE_CUDA

#include <cuda_runtime.h>

// Must define GGML_COMMON_DECL_CUDA before ggml-common.h is first included,
// so that ggml_half = half (CUDA __half) for the kernel inline functions.
#define GGML_COMMON_DECL_CUDA
#  include "../ggml/src/ggml-cuda/ggml-cuda-moe-graph.cuh"
#undef GGML_COMMON_DECL_CUDA

void llama_moe_l1_promote_to_q8_0(
    const void * src_cpu,
    struct ggml_tensor * l1_tensor,
    size_t l1_slot_offset,
    int64_t n_elems,
    ggml_type src_type)
{
    if (!src_cpu || !l1_tensor || !l1_tensor->data) {
        return;
    }

    // GPU-path: copy L2 source to device then run dequant+requant kernel.
    // This is 1 PCIe transfer (L2 source) instead of 1 CPU quant + 1 PCIe
    // transfer (Q8_0 output), saving ~2x bandwidth.
    void * dst_gpu = (char *)l1_tensor->data + l1_slot_offset;
    moe_promote_to_q8_0(src_cpu, dst_gpu, n_elems, src_type, 0); // default stream
}

#else

void llama_moe_l1_promote_to_q8_0(
    const void *,
    struct ggml_tensor *,
    size_t,
    int64_t,
    ggml_type)
{
}

#endif // GGML_USE_CUDA
