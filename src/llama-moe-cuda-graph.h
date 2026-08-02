#pragma once

#include "ggml.h"

#include <vector>

// Per-layer state for MoE callback shortcut.
// Setting --moe-cuda-graphs enables the eval callback's return-false path,
// which batches FFN tensors into larger sub-graphs for fewer GPU sync points.
struct llama_moe_cuda_graph_context {
    size_t n_layers = 0;
    bool   any_captured = false; // set to true when flags are initialized

    // Per-layer flags: true for layers where the callback should skip ggml compute
    std::vector<bool> layer_skip_experts;

    // Per-layer flag: reset to false at start of each layer's gate callback,
    // set to true after the first expert-path tensor is seen
    std::vector<bool> layer_replay_succeeded;

    // Data pointers of FFN norm output tensors per layer (captured from ask=false)
    std::vector<const void *> layer_norm_ptrs;

    // Data pointers of residual input tensors per layer (captured from ask=false)
    std::vector<const void *> layer_residual_ptrs;

    bool is_active() const { return any_captured; }
};

// Initialize per-layer flags for --moe-cuda-graphs mode.
// n_layers: total decoder layers to enable.
bool llama_moe_cuda_graph_capture(
    llama_moe_cuda_graph_context & ctx,
    size_t                         n_layers);

// Reset flags on shutdown.
void llama_moe_cuda_graph_destroy(
    llama_moe_cuda_graph_context & ctx);

// Upload L2 slot quantized data to GPU, convert to Q8_0 directly into L1 GPU tensor.
// l1_tensor must be a CUDA buffer.
void llama_moe_l1_promote_to_q8_0(
    const void * src_cpu,
    struct ggml_tensor * l1_tensor,
    size_t l1_slot_offset,
    int64_t n_elems,
    ggml_type src_type);
