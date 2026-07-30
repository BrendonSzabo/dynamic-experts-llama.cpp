#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// dyn-ex: .bin file reader for VLLM\x02 expert weight format (produced by convert-gguf-to-expert-binary.py)

#define DYN_EX_MAGIC     "VLLM\x02"
#define DYN_EX_VERSION   2
#define DYN_EX_HEADER_SIZE 16384

struct dyn_ex_param {
    char     name[256];
    int64_t  shape[4];
    int      ndim;
    uint8_t  dtype_code;   // from convert-gguf-to-expert-binary.py mapping
    enum ggml_type type;    // resolved ggml type
};

struct dyn_ex_reader {
    int      fd;
    void *   mmap_addr;
    size_t   mmap_size;

    int      n_layers;
    int      n_experts;
    int64_t  expert_stride;  // total bytes per expert (sum of all params, rounded up)
    int      n_params;

    dyn_ex_param params[8];        // up to 8 params
    size_t       param_data_off[8]; // file offset where each param's data section starts
    size_t       param_stride[8];   // per-expert byte stride for each param (page-aligned)
};

// open and mmap a .bin file, parse header. returns nullptr on failure.
dyn_ex_reader * dyn_ex_reader_open(const char * path);

// close and free
void dyn_ex_reader_close(dyn_ex_reader * r);

// read one expert's weights for one param into a caller-provided buffer.
// returns actual bytes read, or 0 on error.
size_t dyn_ex_read_param(const dyn_ex_reader * r, int param_idx, int layer, int expert_id,
                         void * buf, size_t buf_size);

// get per-expert byte size for a param
size_t dyn_ex_param_size(const dyn_ex_reader * r, int param_idx);

// find param index by name (e.g. "gate_up_proj", "down_proj"). returns -1 if not found.
int dyn_ex_param_index(const dyn_ex_reader * r, const char * name);

// ── slot cache ──────────────────────────────────────────────────────────

#define DYN_EX_SENTINEL (-1)

struct dyn_ex_cache {
    dyn_ex_reader * reader;

    int n_layers;
    int n_experts;
    int n_slots;

    // param indices in the reader
    int pi_gate_up;  // "gate_up_proj" or -1
    int pi_gate;     // "gate_proj" or -1
    int pi_up;       // "up_proj" or -1
    int pi_down;     // "down_proj"

    // per-expert and per-slot byte sizes
    size_t gate_up_expert_size; // bytes per expert for fused gate+up
    size_t gate_expert_size;    // bytes per expert for separate gate
    size_t up_expert_size;      // bytes per expert for separate up
    size_t down_expert_size;

    // GPU slot buffers
    ggml_backend_buffer_ptr buf_gate_up; // [n_layers, n_slots, gate_up_expert_size] (merged)
    ggml_backend_buffer_ptr buf_gate;    // [n_layers, n_slots, gate_expert_size] (separate)
    ggml_backend_buffer_ptr buf_up;      // [n_layers, n_slots, up_expert_size]   (separate)
    ggml_backend_buffer_ptr buf_down;    // [n_layers, n_slots, down_expert_size]
    ggml_backend_buffer_ptr buf_slot_map;

    // GPU tensors pointing into the buffers
    struct ggml_tensor * slot_gate_up = nullptr; // shape [..., n_slots] for ggml_mul_mat_id
    struct ggml_tensor * slot_down    = nullptr; // shape [..., n_slots] for ggml_mul_mat_id
    struct ggml_tensor * slot_map     = nullptr; // [n_expert] int32 per layer (flat: n_layers * n_expert)

    // host-side mirror of slot_map (written by ensure, sync'd to GPU)
    std::vector<int32_t> h_slot_of;   // [n_layers * n_expert], DYN_EX_SENTINEL = not present
    std::vector<int32_t> h_expert_in; // [n_layers * n_slots], DYN_EX_SENTINEL = empty
    std::vector<uint8_t>  h_slot_used;// [n_layers * n_slots]

    // async prefetch infrastructure
    ggml_backend_t                  copy_backend = nullptr; // separate backend for async H→D copies
    std::vector<ggml_backend_event_t> copy_events;  // [n_layers * n_slots], per-slot events
    std::vector<std::vector<uint8_t>> staging_bufs; // pinned CPU staging buffers (ring)
    int                             staging_idx = 0;       // current staging buffer index

    // per-layer slot tensors (needed for async copies via ggml_backend_tensor_set_async)
    std::vector<ggml_tensor *> t_gate_up; // [n_layers], per-layer slot tensors for gate_up
    std::vector<ggml_tensor *> t_gate;    // [n_layers], per-layer slot tensors for gate (separate)
    std::vector<ggml_tensor *> t_up;      // [n_layers], per-layer slot tensors for up (separate)
    std::vector<ggml_tensor *> t_down;    // [n_layers], per-layer slot tensors for down

    // per-layer selected_experts capture buffers (GPU→CPU readback after graph compute)
    std::vector<ggml_backend_buffer_ptr> buf_se_capture; // [n_layers]
    std::vector<ggml_tensor *>           t_se_capture;   // [n_layers] I32, [n_expert_used, n_tokens]
    std::vector<ggml_backend_buffer_ptr> buf_barrier; // per-layer barrier buffers (host-visible)
    std::vector<ggml_tensor *>           t_barrier;   // per-layer barrier tensors
    std::vector<void *>                  t_barrier_host; // host pointers for CPU access

    size_t gate_up_stride = 0; // bytes per slot in gate_up buffer (aligned)
    size_t gate_stride    = 0; // bytes per slot in gate buffer (separate, aligned)
    size_t up_stride      = 0; // bytes per slot in up buffer (separate, aligned)
    size_t down_stride    = 0; // bytes per slot in down buffer (aligned)
};

// create slot cache backed by .bin file. returns nullptr on failure.
// dev: GPU device for slot buffers
// n_slots: number of slots per layer (must be power of 2)
// expert_tensor_gate_up: the original ggml tensor for gate_up_exps (provides shape/dtype info)
// expert_tensor_down:    the original ggml tensor for down_exps
dyn_ex_cache * dyn_ex_cache_init(
    struct dyn_ex_reader * reader,
    int n_slots,
    ggml_backend_dev_t dev,
    struct ggml_tensor * expert_gate_up,  // can be nullptr (for separate gate/up models)
    struct ggml_tensor * expert_gate,     // can be nullptr
    struct ggml_tensor * expert_up,       // can be nullptr
    struct ggml_tensor * expert_down);

void dyn_ex_cache_free(struct dyn_ex_cache * cache);

// ensure expert_ids are loaded into slots for the given layer (blocking).
// updates h_slot_of / h_expert_in / h_slot_used, syncs slot_map to GPU.
void dyn_ex_cache_ensure(struct dyn_ex_cache * cache, int layer, const int * expert_ids, int n_ids);

// fill initial slots with first n_slots experts of each layer
void dyn_ex_cache_fill(struct dyn_ex_cache * cache);
void dyn_ex_cache_alloc_barriers(struct dyn_ex_cache * cache, ggml_backend_dev_t dev, int n_layers, int n_expert_used);

// async prefetch: start loading expert_ids into slots without blocking.
// scores: optional (n_expert) array for eviction priority (higher = keep). may be nullptr.
void dyn_ex_cache_prefetch(struct dyn_ex_cache * cache, int layer, const int * expert_ids, int n_ids,
                           const float * scores);

// wait for all in-flight async copies on a given layer to complete
void dyn_ex_cache_wait(struct dyn_ex_cache * cache, int layer);

// ── predictor ──────────────────────────────────────────────────────────

struct dyn_ex_predictor {
    int D; // hidden size
    int L; // n_layers
    int E; // n_experts
    int H; // hidden dim (default 16)

    // weights (CPU, loaded from file)
    std::vector<float> trunk_w; // [H, D]
    std::vector<float> trunk_b; // [H]
    std::vector<float> W1;      // [L*H*(H+E)]
    std::vector<float> b1;      // [L*H]
    std::vector<float> W2;      // [L*E*H]
    std::vector<float> b2;      // [L*E]

    // AdamW optimizer state
    std::vector<float> m_trunk_w, v_trunk_w;
    std::vector<float> m_trunk_b, v_trunk_b;
    std::vector<float> m_W1, v_W1, m_b1, v_b1;
    std::vector<float> m_W2, v_W2, m_b2, v_b2;
    int step = 0; // optimizer step counter
};

// load predictor weights from a binary file.
// file format: 16-byte header (magic "DXP2\0", D, L, E, H as int32), then weights as float32 in order.
dyn_ex_predictor * dyn_ex_predictor_load(const char * path, int D, int L, int E, int H);
void dyn_ex_predictor_free(dyn_ex_predictor * p);

// predict top-k expert IDs for the next token at a given layer.
// ht: hidden states [n_tokens * D], CPU float32
// Et: current expert mask [n_tokens * E], CPU float32
// layer: current layer index (0..L-1)
// top_k: number of experts to predict
// top_ids: output [n_tokens * top_k], int32
// scores: output [n_tokens * E], float32 (may be nullptr)
void dyn_ex_predictor_predict(dyn_ex_predictor * p,
    int n_tokens,
    const float * ht, const float * Et, int layer, int top_k,
    int * top_ids, float * scores);
