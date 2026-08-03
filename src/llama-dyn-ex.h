#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// dyn-ex: .bin file reader for VLLM\x02 expert weight format (produced by convert-gguf-to-expert-binary.py)

#define DYN_EX_MAGIC     "VLLM\x02"
#define DYN_EX_VERSION   2
#define DYN_EX_HEADER_SIZE 16384

struct dyn_ex_param {
    char     name[256];
    int64_t  shape[4];
    int      ndim;
    uint8_t  dtype_code;
    enum ggml_type type;
};

struct dyn_ex_reader {
    int      fd;
    void *   mmap_addr;
    size_t   mmap_size;

    int      n_layers;
    int      n_experts;
    int64_t  expert_stride;
    int      n_params;

    dyn_ex_param params[8];
    size_t       param_data_off[8];
    size_t       param_stride[8];
    size_t       param_size[8];
};

dyn_ex_reader * dyn_ex_reader_open(const char * path);
void dyn_ex_reader_close(dyn_ex_reader * r);

size_t dyn_ex_read_param(const dyn_ex_reader * r, int param_idx, int layer, int expert_id,
                         void * buf, size_t buf_size);
size_t dyn_ex_param_size(const dyn_ex_reader * r, int param_idx);
int dyn_ex_param_index(const dyn_ex_reader * r, const char * name);

// ── 3-level expert cache: L1 (GPU, global) → L2 (host, per-layer) → L3 (.bin mmap) ──

#define DYN_EX_SENTINEL (-1)

struct dyn_ex_cache {
    dyn_ex_reader * reader;

    int n_l1;          // GPU slots (--dyn-ex-l1)
    int n_l2;          // host slots per layer (--dyn-ex-l2)
    int n_layers;
    int n_experts;
    int n_expert_used;
    int n_ubatch;      // max(1, n_l1 / n_expert_used / 2)
    int n_hot;         // n_ubatch * n_expert_used — active range [0, n_hot)
    int n_cache;       // n_l1 - n_hot — LRU cache range [n_hot, n_l1)

    // .bin param indices
    int pi_gate_up, pi_gate, pi_up, pi_down;

    // L1: global GPU tensors (all layers share these)
    struct ggml_tensor * l1_gate    = nullptr;
    struct ggml_tensor * l1_up      = nullptr;
    struct ggml_tensor * l1_down_q4 = nullptr;
    struct ggml_tensor * l1_down_q6 = nullptr;
    struct ggml_tensor * l1_gate_up = nullptr;

    // per-slot stride for L1 (max across layers, with zero-padding for smaller)
    size_t l1_stride_gate    = 0;
    size_t l1_stride_up      = 0;
    size_t l1_stride_down    = 0;
    size_t l1_stride_gate_up = 0;

    // L1 slot tracking [n_l1]
    std::vector<int32_t> l1_layer;   // which layer, -1 = empty
    std::vector<int32_t> l1_expert;  // which expert
    std::vector<uint64_t> l1_age;    // LRU timestamp
    std::vector<uint8_t>  l1_in_use; // 1 = pinned (for future features)

    // L2: per-layer host buffers [n_layers]
    struct l2_layer {
        std::vector<uint8_t> gate;
        std::vector<uint8_t> up;
        std::vector<uint8_t> down;
        std::vector<uint8_t> gate_up;
        std::vector<int32_t>  expert;        // [n_l2] which expert in each slot, -1 = empty
        std::vector<int32_t>  slot_of;       // [n_experts] which slot holds each expert, -1 = not cached
        std::vector<uint64_t> age;           // [n_l2] LRU
        size_t gate_size    = 0;
        size_t up_size      = 0;
        size_t down_size    = 0;
        size_t gate_up_size = 0;
        size_t gate_row     = 0;
        size_t up_row       = 0;
        size_t down_row     = 0;
        size_t gate_up_row  = 0;
    };
    std::vector<l2_layer> l2;

    uint64_t clock = 0;

    // barriers per layer
    std::vector<struct ggml_tensor *> t_barrier;
    std::vector<void *>               t_barrier_host;

#ifdef GGML_USE_CUDA
    std::mutex l2_mutex; // protects L2 eviction
#endif
};

dyn_ex_cache * dyn_ex_cache_init(
    struct dyn_ex_reader * reader,
    int n_l1, int n_l2, int n_expert_used,
    ggml_backend_dev_t dev);

void dyn_ex_cache_free(struct dyn_ex_cache * cache);
void dyn_ex_cache_alloc_barriers(struct dyn_ex_cache * cache, int n_layers, int n_expert_used);

// set per-layer expert size info for L2 sizing (called from dyn_ex_init after tensor creation)
void dyn_ex_cache_set_layer_size(
    struct dyn_ex_cache * cache, int layer,
    size_t gate_size, size_t gate_row, size_t up_size, size_t up_row,
    size_t down_size, size_t down_row, size_t gate_up_size, size_t gate_up_row);

// ── predictor (unchanged) ──

struct dyn_ex_predictor {
    int D, L, E, H;
    std::vector<float> trunk_w, trunk_b, W1, b1, W2, b2;
    std::vector<float> m_trunk_w, v_trunk_w, m_trunk_b, v_trunk_b;
    std::vector<float> m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2;
    int step = 0;
};

dyn_ex_predictor * dyn_ex_predictor_load(const char * path, int D, int L, int E, int H);
void dyn_ex_predictor_free(dyn_ex_predictor * p);
void dyn_ex_predictor_predict(dyn_ex_predictor * p,
    int n_tokens, const float * ht, const float * Et, int layer, int top_k,
    int * top_ids, float * scores);
