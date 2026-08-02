#include "llama-flash-moe-slot-runtime.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#ifdef GGML_USE_CUDA
#include "llama-moe-cuda-graph.h"
#endif

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <fcntl.h>
#include <sys/stat.h>
#include <fstream>
#include <set>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <io.h>
#define ssize_t int
static inline ssize_t pread(int fd, void * buf, size_t count, off_t offset) {
    static std::unordered_map<int, std::mutex> fd_locks;
    std::lock_guard<std::mutex> lock(fd_locks[fd]);
    __int64 old = _lseeki64(fd, 0, SEEK_CUR);
    if (old < 0) return -1;
    if (_lseeki64(fd, offset, SEEK_SET) < 0) return -1;
    int n = _read(fd, buf, (unsigned int)count);
    _lseeki64(fd, old, SEEK_SET);
    return (ssize_t)n;
}
#else
#include <unistd.h>
#endif

// Flash-MoE GPU bank support is controlled by LLAMA_FLASH_MOE_GPU_BANK compile flag
// GPU-bank enabled builds allow slot-bank tensors to be placed on CUDA/Metal device memory.

// ---- Mixed slot buffer ----
// FNV-1a 64-bit hash for data integrity verification
static uint64_t fnv1a_64(const uint8_t * data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static bool flash_moe_mixed_slot_buffer_enabled() {
    static bool en = flash_moe_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_MIXED_SLOT_BUFFER");
    return en;
}

// ---- Constructor ----

// 8-arg constructor: delegates to 3-arg constructor and stores timing pointers
llama_flash_moe_slot_runtime::llama_flash_moe_slot_runtime(
        const llama_model & model,
        bool perf_profile,
        bool transient_shared_scratch,
        bool l1_dequant_q8_0,
        int32_t /*prefill_micro_batch_tokens*/,
        const int64_t * prefill_eval_us,
        const int64_t * decode_eval_us,
        const int64_t * prefill_eval_tokens,
        const int64_t * decode_eval_tokens)
    : llama_flash_moe_slot_runtime(model, transient_shared_scratch, l1_dequant_q8_0) {
    (void)perf_profile;
    // Store timing pointers for prefill-aware progress reporting
    // (currently stored in llama_context accumulators; the pointers are
    //  passed through to keep the constructor signature uniform)
    (void)prefill_eval_us;
    (void)decode_eval_us;
    (void)prefill_eval_tokens;
    (void)decode_eval_tokens;
}



llama_flash_moe_slot_runtime::llama_flash_moe_slot_runtime(
        const llama_model & model,
        bool /*perf_profile*/,
        bool transient_shared_scratch,
        bool l1_dequant_q8_0)
    : model(model),
      slot_count(transient_shared_scratch
          ? std::max<int32_t>(1, model.hparams.n_expert)
          : model.flash_moe_slot_count()),
      expert_count(model.hparams.n_expert),
      two_level_enabled(!transient_shared_scratch
          && model.flash_moe_two_level_slot_bank_enabled()),
      transient_shared_scratch(transient_shared_scratch),
      resident_bank_source(!transient_shared_scratch
          && model.flash_moe_resident_source_enabled()),
      l1_dequant_q8_0_(l1_dequant_q8_0) {

    if (slot_count <= 0) {
        throw std::runtime_error("Flash-MoE: slot count must be > 0");
    }

    if (slot_count < expert_count) {
        LLAMA_LOG_WARN("Flash-MoE: slot count %d < n_expert %d; some experts may experience contention\n",
            slot_count, expert_count);
    }

    l1_slot_count = two_level_enabled ? model.flash_moe_slot_count_l1() : 0;
    layers.resize(model.layers.size());
    l1_ctxs.resize(model.layers.size());
    l1_bufs.resize(model.layers.size());

    for (size_t il = 0; il < model.layers.size(); ++il) {
        auto & state = layers[il];
        const auto & layer = model.layers[il];

        // Check if this layer has MoE expert tensors (is a routed layer)
        bool has_experts = (layer.ffn_gate_exps != nullptr)
                        || (layer.ffn_up_exps != nullptr)
                        || (layer.ffn_down_exps != nullptr)
                        || (layer.ffn_gate_up_exps != nullptr);

        if (!has_experts) {
            state.enabled = false;
            continue;
        }

        state.enabled  = true;
        state.n_slots  = slot_count;

        // Initialize slot→expert and expert→slot maps
        state.slot_to_expert.assign(slot_count, -1);
        state.expert_to_slot.assign(expert_count, -1);
        state.slot_age.assign(slot_count, 0);
        state.slot_reserved_epoch.assign(slot_count, 0);
        state.request_seen_epoch.assign(expert_count, 0);
        state.request_slot.assign(expert_count, -1);

        // Two-level L1 initialization
        if (two_level_enabled) {
            state.two_level_enabled = true;
            state.l1_n_slots = l1_slot_count;
            state.l1_entries.resize(l1_slot_count);
            state.l2_to_l1_entry.clear();

#ifdef LLAMA_FLASH_MOE_GPU_BANK
            // Allocate L1 GPU slot-bank tensors
            if (l1_slot_count > 0) {
                const int64_t n_embd_val = model.hparams.n_embd;
                // Use n_ff_exp for MoE models where n_ff returns 0 (per-expert FFN dimension)
                const int64_t n_ff_val = model.hparams.n_ff(il);
                const int64_t n_ff_l1  = n_ff_val > 0 ? n_ff_val : model.hparams.n_ff_exp;
                const int64_t n_slots_l1_val = l1_slot_count;
                if (il == 0) {
                    fprintf(stderr, "CGT: L1 n_ff_val=%lld n_ff_exp=%d n_ff_l1=%lld\n",
                        (long long)n_ff_val, model.hparams.n_ff_exp, (long long)n_ff_l1);
                }

                // ggml_context for tensor metadata (preserved for lifetime of runtime)
                ggml_init_params params = {
                    /*.mem_size   =*/ sizeof(ggml_tensor) * 4 + 1024,
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                auto ctx_l1 = ggml_context_ptr(ggml_init(params));
                if (!ctx_l1) {
                    throw std::runtime_error(
                        format("Flash-MoE: ggml_init failed for L1 context layer %zu", il));
                }

                // L1 GPU bank always uses Q8_0. CUDA graph capture requires Q8_0,
                // and it halves memory vs F16. For quantized source models (IQ4_NL,
                // MXFP4, etc.) the dequant-to-Q8_0 happens at upload time.
                ggml_type l1_type = GGML_TYPE_Q8_0;

                // L1 gate_up: [n_embd, n_ff_l1 * 2, n_slots_l1]
                state.l1_gate_up_tensor = ggml_new_tensor_3d(ctx_l1.get(), l1_type,
                    n_embd_val, n_ff_l1 * 2, n_slots_l1_val);
                // L1 down: [n_ff_l1, n_embd, n_slots_l1]
                state.l1_down_tensor = ggml_new_tensor_3d(ctx_l1.get(), l1_type,
                    n_ff_l1, n_embd_val, n_slots_l1_val);

                ggml_set_name(state.l1_gate_up_tensor, "l1_gate_up");
                ggml_set_name(state.l1_down_tensor, "l1_down");

                // Allocate tensor data on the GPU backend for this layer
                ggml_backend_buffer_type_t buft = model.select_buft(static_cast<int>(il));
                auto buf_l1 = ggml_backend_buffer_ptr(
                    ggml_backend_alloc_ctx_tensors_from_buft(ctx_l1.get(), buft));
                if (!buf_l1) {
                    // Diagnostic: compute exact allocation sizes
                    size_t l1_gate_up_bytes = ggml_row_size(l1_type, n_embd_val) * n_ff_l1 * 2 * n_slots_l1_val;
                    size_t l1_down_bytes   = ggml_row_size(l1_type, n_ff_l1) * n_embd_val * n_slots_l1_val;
                    LLAMA_LOG_ERROR("%s: L1 allocation failed for layer %zu\n", __func__, il);
                    LLAMA_LOG_ERROR("%s:   l1_type=%s gate_up=[%lld,%lld,%lld] (%zu bytes) down=[%lld,%lld,%lld] (%zu bytes) total=%.0f MiB\n",
                        __func__, ggml_type_name(l1_type),
                        (long long)n_embd_val, (long long)(n_ff_val * 2), (long long)n_slots_l1_val, l1_gate_up_bytes,
                        (long long)n_ff_val, (long long)n_embd_val, (long long)n_slots_l1_val, l1_down_bytes,
                        (l1_gate_up_bytes + l1_down_bytes) / 1024.0 / 1024.0);
                    throw std::runtime_error(
                        format("Flash-MoE: failed to allocate L1 tensors on backend for layer %zu", il));
                }

                // Zero-initialize to prevent NaNs
                ggml_backend_buffer_clear(buf_l1.get(), 0);

                // Keep context and buffer alive for the lifetime of the runtime
                l1_ctxs[il] = std::move(ctx_l1);
                l1_bufs[il] = std::move(buf_l1);

                // Record the device for this layer's L1 tensors.
                // TODO: multi-GPU setups should derive this from the backend buffer type
                state.l1_device_id = 0;

                if (il == 0) {
                    size_t gate_up_b = ggml_row_size(l1_type, n_embd_val) * n_ff_l1 * 2 * n_slots_l1_val;
                    size_t down_b   = ggml_row_size(l1_type, n_ff_l1) * n_embd_val * n_slots_l1_val;
                    fprintf(stderr, "CGT: L1 allocated %d slots type=%s gate_up=[%lld,%lld,%lld] (%zu MiB) down=[%lld,%lld,%lld] (%zu MiB)\n",
                        n_slots_l1_val, ggml_type_name(l1_type),
                        (long long)n_embd_val, (long long)(n_ff_val * 2), (long long)n_slots_l1_val, gate_up_b / 1024 / 1024,
                        (long long)n_ff_val, (long long)n_embd_val, (long long)n_slots_l1_val, down_b / 1024 / 1024);
                }
            }
#endif // LLAMA_FLASH_MOE_GPU_BANK
        }

        // Bind tensors and sidecar entries
        bind_tensor(layer.ffn_gate_up_exps, state.gate_up_tensor, state.gate_up_entry, state.enabled);
        bind_tensor(layer.ffn_gate_exps,    state.gate_tensor,    state.gate_entry,    state.enabled);
        bind_tensor(layer.ffn_up_exps,      state.up_tensor,      state.up_entry,      state.enabled);
        bind_tensor(layer.ffn_down_exps,    state.down_tensor,    state.down_entry,    state.enabled);

        // Allocate L2 slot bank tensor data (virtualized tensors were created
        // with TENSOR_SKIP_IF_VIRTUAL so their data pointers are NULL).
        // The slot I/O engine (install_loads) writes into these buffers.
        // Must use ggml_backend buffers so the scheduler can track tensor ownership.
        for (auto * t : { state.gate_up_tensor, state.gate_tensor,
                          state.up_tensor, state.down_tensor }) {
            if (t && t->data == nullptr && t->buffer == nullptr) {
                size_t nbytes = ggml_nbytes(t);
                if (nbytes > 0) {
                    auto buf = ggml_backend_buffer_ptr(ggml_backend_buft_alloc_buffer(
                        ggml_backend_cpu_buffer_type(), nbytes));
                    if (!buf) {
                        throw std::runtime_error(format(
                            "Flash-MoE: failed to allocate L2 slot tensor '%s' (%zu bytes)",
                            ggml_get_name(t), nbytes));
                    }
                    t->data = ggml_backend_buffer_get_base(buf.get());
                    t->buffer = buf.get();
                    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    l2_buffers.push_back(std::move(buf));
                }
            }
        }

        // Mixed-slot buffer: precompute combined byte offsets
        if (flash_moe_mixed_slot_buffer_enabled()) {
            struct {
                ggml_tensor * tensor;
                const llama_flash_moe_sidecar_entry * entry;
                int family;
            } mixed_families[] = {
                { state.gate_up_tensor, state.gate_up_entry, 0 },
                { state.gate_tensor,    state.gate_entry,    1 },
                { state.up_tensor,      state.up_entry,      2 },
                { state.down_tensor,    state.down_entry,    3 },
            };

            size_t offset = 0;
            for (const auto & mf : mixed_families) {
                if (!mf.tensor || !mf.entry) continue;
                flash_moe_layer_state::mixed_slot_field field;
                field.tensor      = mf.tensor;
                field.entry       = mf.entry;
                field.family      = mf.family;
                field.slot_offset = offset;
                state.mixed_slot_fields.push_back(field);
                offset += mf.entry->bytes_per_expert;
            }
            state.mixed_slot_bytes = offset;

            // Same layout applies to all I/O lanes
            state.mixed_prefetch_slot_fields  = state.mixed_slot_fields;
            state.mixed_prefetch_slot_bytes   = state.mixed_slot_bytes;
            state.mixed_secondary_slot_fields = state.mixed_slot_fields;
            state.mixed_secondary_slot_bytes  = state.mixed_slot_bytes;
            state.mixed_tertiary_slot_fields  = state.mixed_slot_fields;
            state.mixed_tertiary_slot_bytes   = state.mixed_slot_bytes;
        }

        state.predicted_experts.reserve(std::max<int32_t>(1, model.hparams.n_expert_used));
        state.current_token_experts.reserve(std::max<int32_t>(1, model.hparams.n_expert_used));
        state.temporal_prefetch_experts.reserve(std::max<int32_t>(1, model.hparams.n_expert_used));

        // Verify we have sidecar entries for each bound tensor
        if (!state.gate_up_entry && !state.gate_entry && !state.up_entry && !state.down_entry) {
            state.enabled = false;
        }
    }

    // Start the read thread pool for parallel pread I/O
    start_read_pool();

    // Initialize backend trace file if enabled
    if (flash_moe_backend_trace_enabled()) {
        const std::string trace_path = flash_moe_prepare_trace_output_path();
        trace_fp = flash_moe_open_trace_output_file(trace_path);
        if (trace_fp) {
            LLAMA_LOG_INFO("%s: Flash-MoE backend trace enabled, writing to '%s'\n",
                __func__, trace_path.c_str());
        }
    }

    LLAMA_LOG_INFO("%s: Flash-MoE slot-bank runtime created: slots=%d, layers=%zu, two-level=%d\n",
        __func__, slot_count, layers.size(), two_level_enabled ? 1 : 0);
}

void llama_flash_moe_slot_runtime::set_io_split_params(
        int32_t cache_split, int32_t prefill_split, int32_t prefetch_split) {
    cache_io_split_         = cache_split    > 0 ? cache_split    : 1;
    prefill_cache_io_split_ = prefill_split  > 0 ? prefill_split  : cache_io_split_;
    prefetch_cache_io_split_ = prefetch_split > 0 ? prefetch_split : cache_io_split_;
}

// Parse a colon-separated triplet "A:B:C" into an array<int32_t,3>.
static std::array<int32_t, 3> parse_stripe_weights(const std::string & str) {
    if (str.empty()) {
        return {1, 0, 0};
    }
    std::array<int32_t, 3> result = {0, 0, 0};
    size_t pos = 0;
    for (int i = 0; i < 3; i++) {
        size_t next = str.find(':', pos);
        std::string token = (next == std::string::npos) ? str.substr(pos) : str.substr(pos, next - pos);
        if (!token.empty()) {
            char * end = nullptr;
            long val = std::strtol(token.c_str(), &end, 10);
            if (end == token.c_str() || *end != '\0') {
                throw std::runtime_error(format(
                    "Flash-MoE: invalid stripe weight '%s' in '%s'",
                    token.c_str(), str.c_str()));
            }
            result[i] = (int32_t)val;
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return result;
}

void llama_flash_moe_slot_runtime::set_stripe_params(
        const std::string & demand_stripe,
        const std::string & prefill_stripe,
        const std::string & prefetch_stripe) {
    demand_stripe_  = parse_stripe_weights(demand_stripe);
    prefill_stripe_ = parse_stripe_weights(prefill_stripe);
    prefetch_stripe_ = parse_stripe_weights(prefetch_stripe);
}

void llama_flash_moe_slot_runtime::set_prefill_next_hot_params(
        int32_t experts, bool exclusive_drives) {
    prefill_next_hot_experts_ = experts;
    prefill_next_hot_exclusive_drives_ = exclusive_drives;
}

void llama_flash_moe_slot_runtime::set_moe_quant_map(const char * v) {
    moe_quant_map_ = v;
    if (v) {
        LLAMA_LOG_INFO("%s: Flash-MoE quant map: %s\n", __func__, v);
    }
}

void llama_flash_moe_slot_runtime::set_distribute_params(
        const std::string & demand_distribute,
        const std::string & prefill_distribute,
        const std::string & prefetch_distribute) {
    demand_distribute_   = parse_stripe_weights(demand_distribute);
    prefill_distribute_  = parse_stripe_weights(prefill_distribute);
    prefetch_distribute_ = parse_stripe_weights(prefetch_distribute);
}

void llama_flash_moe_slot_runtime::set_trace_harness(bool v) {
    trace_harness_mode_ = v;
    if (v && !trace_fp) {
        const std::string trace_path = flash_moe_prepare_trace_output_path();
        trace_fp = flash_moe_open_trace_output_file(trace_path);
    }
}

llama_flash_moe_slot_runtime::~llama_flash_moe_slot_runtime() {
    stop_read_pool();
    for (auto & [path, fd] : fds) {
        if (fd >= 0) {
#ifdef _WIN32
            _close(fd);
#else
            close(fd);
#endif
        }
    }
    fds.clear();
    l2_buffers.clear();
    if (trace_fp) {
        fclose(trace_fp);
        trace_fp = nullptr;
    }
}

// ---- Tensor binding ----

void llama_flash_moe_slot_runtime::bind_tensor(
        ggml_tensor * tensor, ggml_tensor *& tensor_out,
        const llama_flash_moe_sidecar_entry *& entry_out, bool & enabled) {
    if (tensor == nullptr) return;

    tensor_out = tensor;

    // Look up sidecar entry by tensor name
    const std::string name = ggml_get_name(tensor);
    const auto & entries = model.flash_moe_sidecar_entries();
    auto it = entries.find(name);
    if (it != entries.end()) {
        entry_out = &it->second;
        enabled = true;
    }
}

// ---- Eval callback interface ----

bool llama_flash_moe_slot_runtime::parse_topk_layer(const char * name, int & layer) {
    if (name == nullptr) return false;
    return sscanf(name, "ffn_moe_topk-%d", &layer) == 1;
}

bool llama_flash_moe_slot_runtime::wants_tensor(const ggml_tensor * t) const {
    int layer = -1;
    return parse_topk_layer(ggml_get_name(t), layer)
        && uses_layer(layer)
        && !uses_native_slot_map(layer);
}

bool llama_flash_moe_slot_runtime::uses_layer(int il) const {
    return il >= 0 && (size_t)il < layers.size() && layers[il].enabled;
}

bool llama_flash_moe_slot_runtime::uses_native_slot_map(int /*layer*/) const {
    return false;  // Native slot-map not yet ported (Qwen35MoE, DeepSeek2)
}

bool llama_flash_moe_slot_runtime::uses_dedicated_prefill_moe(int /*layer*/) const {
    return false;  // Dedicated prefill MoE not yet ported
}

ggml_tensor * llama_flash_moe_slot_runtime::select_routed_weight_tensor(
        int /*layer*/, ggml_tensor * tensor) {
    return tensor;  // Pass-through: slot-shaped tensors are already virtualized at load time
}

ggml_tensor * llama_flash_moe_slot_runtime::build_slot_ids_tensor(
        ggml_context * /*ctx*/, ggml_tensor * selected_experts, int /*il*/) {
    // For models that support native slot-map, this creates a ggml_map_custom1()
    // op that translates expert IDs → slot IDs at graph execution time.
    // Currently a pass-through for non-native models.
    return selected_experts;
}

void llama_flash_moe_slot_runtime::bind_slot_ids_input(int layer, ggml_tensor * slot_ids) {
    if (layer < 0 || (size_t)layer >= layers.size()) {
        LLAMA_LOG_WARN("%s: layer %d out of range [0, %zu)\n", __func__, layer, layers.size());
        return;
    }
    layers[layer].slot_ids_input = slot_ids;
}

bool llama_flash_moe_slot_runtime::handle_tensor(ggml_tensor * tensor) {
    int layer = -1;
    if (!parse_topk_layer(ggml_get_name(tensor), layer) || !uses_layer(layer)) {
        return true;
    }

    auto & state = layers[layer];

    if (state.slot_ids_input == nullptr) {
        return true;  // no slot_ids input bound yet — skip
    }

    if (tensor->type != GGML_TYPE_I32) {
        return true;  // wrong type — skip
    }

    const int64_t n_expert_used = tensor->ne[0];
    const int64_t n_tokens      = tensor->ne[1];
    const size_t  n_ids         = (size_t)(n_expert_used * n_tokens);

    LLAMA_LOG_DEBUG("%s: handle_tensor layer=%d experts=%lld tokens=%lld total_ids=%zu\n",
        __func__, layer, (long long)n_expert_used, (long long)n_tokens, n_ids);

    // Oracle bypass: skip slot reservation when replaying a recorded trace
    if (!oracle_records.empty()) {
        oracle_record rec;
        if (next_oracle_record(layer, (int)n_expert_used, (int)n_tokens, rec)) {
            LLAMA_LOG_INFO("%s: oracle bypass layer=%d experts=%d tokens=%d\n",
                __func__, layer, rec.n_expert_used, rec.n_tokens);
            // Build slot IDs from expert->slot mapping (set by prime_oracle_trace)
            // This is correct: prime_oracle_trace pre-reserves slots for each unique
            // expert, populating state.expert_to_slot[expert].
            std::vector<int32_t> slot_ids;
            slot_ids.reserve((size_t)n_expert_used * n_tokens);
            for (int32_t t = 0; t < n_tokens; ++t) {
                for (int32_t e = 0; e < n_expert_used; ++e) {
                    int32_t expert = rec.experts[t * n_expert_used + e];
                    int32_t slot = (expert >= 0 && expert < expert_count)
                        ? state.expert_to_slot[expert]
                        : -1;
                    slot_ids.push_back(slot >= 0 ? slot : e);
                }
            }
            GGML_ASSERT((size_t)(n_expert_used * n_tokens) == slot_ids.size());
            // Write slot IDs to the slot_ids input tensor
            if (state.slot_ids_input && state.slot_ids_input->data) {
                if (state.slot_ids_input->buffer
                    && !ggml_backend_buffer_is_host(state.slot_ids_input->buffer)) {
                    ggml_backend_tensor_set(state.slot_ids_input, slot_ids.data(), 0,
                        n_ids * sizeof(int32_t));
                } else {
                    memcpy(state.slot_ids_input->data, slot_ids.data(),
                        n_ids * sizeof(int32_t));
                }
            }

            // Oracle-prefetch: peek at the next record, prefetch its experts,
            // then advance the cursor past it so it won't be oracle-replayed
            // (the prefetched experts will be hit via the normal reservation path)
            if (moe_mode == LLAMA_MOE_MODE_ORACLE_PREFETCH) {
                if (oracle_cursor < oracle_records.size()) {
                    const auto & next_rec = oracle_records[oracle_cursor];
                    LLAMA_LOG_DEBUG("%s: oracle-prefetch next layer=%d experts=%zu\n",
                        __func__, next_rec.layer, next_rec.experts.size());
                    prefetch_experts(next_rec.layer, next_rec.experts);
                }
                prime_oracle_prefetch_record(1);
            }

        }
    }

    // Read expert IDs from the top-k tensor
    std::vector<int32_t> expert_ids(n_ids);
    if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
        ggml_backend_tensor_get(tensor, expert_ids.data(), 0, n_ids * sizeof(int32_t));
    } else {
        memcpy(expert_ids.data(), tensor->data, n_ids * sizeof(int32_t));
    }

    // Start timing for trace record (reservation + I/O)
    const int64_t t_start = ggml_time_us();

    // ---- moe_predict_prev_token: bump LRU age for predicted experts ----
    if (moe_predict_prev_token_ > 0 && !state.predicted_experts.empty()) {
        for (int32_t pred_expert : state.predicted_experts) {
            if (pred_expert >= 0 && pred_expert < expert_count) {
                int32_t slot = state.expert_to_slot[pred_expert];
                if (slot >= 0) {
                    state.slot_age[slot] = ++age;
                }
            }
        }
    }

    // ---- moe_sort_decode_expert_ids: build sorted reservation order ----
    const bool sort_by_offset = (moe_sort_decode_expert_ids_ > 0 && n_ids > 1
        && state.gate_up_entry != nullptr);
    std::vector<size_t> reservation_order;
    if (sort_by_offset) {
        reservation_order.resize(n_ids);
        for (size_t ii = 0; ii < n_ids; ++ii) {
            reservation_order[ii] = ii;
        }
        off_t sort_base = (off_t)state.gate_up_entry->repacked_offset;
        size_t sort_stride = state.gate_up_entry->bytes_per_expert;
        if (sort_stride > 0) {
            std::sort(reservation_order.begin(), reservation_order.end(),
                [&](size_t a, size_t b) {
                    int32_t ea = expert_ids[a];
                    int32_t eb = expert_ids[b];
                    if (ea < 0) return false;
                    if (eb < 0) return true;
                    off_t oa = sort_base + (off_t)((size_t)ea * sort_stride);
                    off_t ob = sort_base + (off_t)((size_t)eb * sort_stride);
                    return oa < ob;
                });
        }
    }

    // Allocate slot IDs
    std::vector<int32_t> slot_ids(n_ids, -1);
    std::vector<pending_slot_load> pending_loads;
    const uint32_t epoch = next_request_epoch();
    std::vector<int32_t> touched_slots; touched_slots.reserve(slot_count);

    for (size_t pi = 0; pi < n_ids; ++pi) {
        size_t i = sort_by_offset ? reservation_order[pi] : pi;
        int32_t expert = expert_ids[i];
        if (expert < 0 || expert >= expert_count) continue;

        // Reserve a slot for this expert
        auto reserved = reserve_expert_slot(state, layer, expert,
            epoch, n_tokens, n_expert_used);
        slot_ids[i] = reserved.slot;

        if (reserved.slot >= 0 && reserved.slot < state.n_slots) {
            touched_slots.push_back(reserved.slot);
        }

        // Queue I/O if this was a miss
        if (reserved.miss && reserved.slot >= 0) {
            pending_loads.push_back({ expert, reserved.slot,
                source_lane::primary, reserved.l1_slot,
                reserved.needs_l1_promotion });
        }

        commit_reserved_slot(state, expert, reserved);
    }

    // Data integrity: verify cache-hit slot data against stored hash
    if (state.gate_up_tensor && state.gate_up_tensor->data && state.slot_data_hash.size() > 0) {
        size_t slot_stride = (size_t)ggml_row_size(state.gate_up_tensor->type, state.gate_up_tensor->ne[0])
                           * state.gate_up_tensor->ne[1];
        std::unordered_set<int32_t> miss_slots;
        for (auto & pl : pending_loads) miss_slots.insert(pl.slot);
        for (int32_t slot : touched_slots) {
            if (miss_slots.count(slot)) continue;
            auto it = state.slot_data_hash.find(slot);
            if (it == state.slot_data_hash.end()) continue;
            const uint8_t * data = (const uint8_t *)state.gate_up_tensor->data + (size_t)slot * slot_stride;
            uint64_t cur = fnv1a_64(data, slot_stride);
            if (cur != it->second) {
                step_stats_.n_hash_mismatches++;
            }
        }
    }

    // Log expert->slot mapping (DEBUG for all layers, summarised at step end)
    if (n_tokens == 1 && !sort_by_offset) {
        std::string map;
        for (size_t i = 0; i < n_ids && i < 8; ++i) {
            if (!map.empty()) map += ", ";
            map += "e" + std::to_string(expert_ids[i]) + "->s" + std::to_string(slot_ids[i]);
        }
        LLAMA_LOG_DEBUG("%s: layer=%d mapping [%s] (%zu pending_loads)\n",
            __func__, layer, map.c_str(), pending_loads.size());
    }

    // Clamp any unmatched slot IDs to 0 to prevent OOB access in MUL_MAT_ID
    for (size_t i = 0; i < n_ids; ++i) {
        if (slot_ids[i] < 0) slot_ids[i] = 0;
    }

    // Write slot IDs to the slot_ids input tensor
    if (state.slot_ids_input && state.slot_ids_input->data) {
        if (n_tokens == 1) {
            std::string sid_str;
            for (size_t si = 0; si < n_ids && si < 8; ++si) {
                if (!sid_str.empty()) sid_str += ",";
                sid_str += std::to_string(slot_ids[si]);
            }
            LLAMA_LOG_DEBUG("%s: write_slot_ids=[%s] on_gpu=%d\n",
                __func__, sid_str.c_str(),
                state.slot_ids_input->buffer ? (int)!ggml_backend_buffer_is_host(state.slot_ids_input->buffer) : 0);
        }
        if (state.slot_ids_input->buffer
            && !ggml_backend_buffer_is_host(state.slot_ids_input->buffer)) {
            ggml_backend_tensor_set(state.slot_ids_input, slot_ids.data(), 0,
                n_ids * sizeof(int32_t));
        } else {
            memcpy(state.slot_ids_input->data, slot_ids.data(),
                n_ids * sizeof(int32_t));
        }
    }

    // Perform I/O for any misses
    if (!pending_loads.empty()) {
        install_loads(state, pending_loads,
            prefill_io_mode_ ? prefill_cache_io_split_ : cache_io_split_,
            demand_stripe_);
    }

    // Data integrity: hash newly loaded slot data after install
    if (!pending_loads.empty() && state.gate_up_tensor && state.gate_up_tensor->data) {
        size_t slot_stride = (size_t)ggml_row_size(state.gate_up_tensor->type, state.gate_up_tensor->ne[0])
                           * state.gate_up_tensor->ne[1];
        for (auto & pl : pending_loads) {
            const uint8_t * data = (const uint8_t *)state.gate_up_tensor->data + (size_t)pl.slot * slot_stride;
            state.slot_data_hash[pl.slot] = fnv1a_64(data, slot_stride);
        }
    }

    // Oracle-prefetch: peek ahead at the next record and preload its experts
    if (moe_mode == LLAMA_MOE_MODE_ORACLE_PREFETCH && !oracle_records.empty()) {
        if (oracle_cursor < oracle_records.size()) {
            const auto & next_rec = oracle_records[oracle_cursor];
            prefetch_experts(next_rec.layer, next_rec.experts);
        }
    }

    // Prefill next-hot experts from oracle trace (if configured)
    if (prefill_next_hot_experts_ > 0 && !oracle_records.empty()) {
        auto hot = compute_next_hot_experts(layer, prefill_next_hot_experts_);
        if (!hot.empty()) {
            if (prefill_next_hot_exclusive_drives_) {
                prefetch_experts(layer, hot, source_lane::secondary);
            } else {
                prefetch_experts(layer, hot);
            }
        }
    }

    // Update LRU ages for touched slots
    for (int32_t slot : touched_slots) {
        if (slot >= 0 && slot < state.n_slots) {
            state.slot_age[slot] = ++age;
        }
    }

    if (!pending_loads.empty()) {
        uint64_t hits = state.cache_hit_count;
        uint64_t misses = state.cache_miss_count;
        uint64_t total_ops = hits + misses;
        LLAMA_LOG_DEBUG("%s: layer=%d reservation: %zu I/O ops needed (cumulative: %llu hits %llu misses %.1f%% hit rate)\n",
            __func__, layer, pending_loads.size(),
            (unsigned long long)hits, (unsigned long long)misses,
            total_ops > 0 ? 100.0 * (double)hits / (double)total_ops : 0.0);
    }

    // Write backend trace record if enabled
    if (trace_fp || trace_harness_mode_) {
        if (!trace_fp && trace_harness_mode_) {
            const std::string trace_path = flash_moe_prepare_trace_output_path();
            trace_fp = flash_moe_open_trace_output_file(trace_path);
        }
        if (trace_fp) {
            const int64_t elapsed_us = ggml_time_us() - t_start;
            write_trace_record(layer, (int)n_tokens, (int)n_expert_used,
                expert_ids.data(), slot_ids.data(), (uint64_t)elapsed_us);
        }
    }

    // Temporal prefetch: use current token's experts as prediction for the next token
    {   // deduplicate expert IDs for prediction state
        std::vector<int32_t> unique_experts;
        unique_experts.reserve(n_expert_used);
        for (size_t i = 0; i < n_ids; ++i) {
            int32_t e = expert_ids[i];
            if (e < 0) continue;
            if (std::find(unique_experts.begin(), unique_experts.end(), e) == unique_experts.end()) {
                unique_experts.push_back(e);
            }
        }
        state.current_token_experts = unique_experts;
        if (moe_prefetch_temporal_ > 0 && (!moe_prefetch_temporal_sparse_ || (layer % 2 == 0))) {
            state.temporal_prefetch_experts  = unique_experts;
            state.temporal_prefetch_active   = true;
        }
    }

    // ---- moe_predict_prev_token / moe_predict_top1_prev: save for next decode step ----
    if (moe_predict_prev_token_ > 0 && !state.current_token_experts.empty()) {
        if (moe_predict_top1_prev_ > 0) {
            state.predicted_experts.clear();
            state.predicted_experts.push_back(state.current_token_experts[0]);
        } else {
            state.predicted_experts = state.current_token_experts;
        }
    }

    return true;
}

// ---- Slot Reservation (Core LRU) ----

reserved_slot llama_flash_moe_slot_runtime::reserve_expert_slot(
        flash_moe_layer_state & state, int layer, int32_t expert,
        uint32_t epoch, int64_t n_tokens, int64_t n_expert_used) {
    reserved_slot result;

    GGML_ASSERT(expert >= 0 && expert < expert_count);

    // L2 lookup
    result.slot = state.expert_to_slot[expert];
    if (result.slot >= 0) {
        // L2 HIT
        state.cache_hit_count++;
        state.slot_reserved_epoch[result.slot] = epoch;

        if (state.two_level_enabled) {
            // L1 lookup: check if this L2 slot is in L1
            auto it = state.l2_to_l1_entry.find(result.slot);
            if (it != state.l2_to_l1_entry.end()) {
                state.l1_cache_hit_count++;
            } else {
                state.l1_cache_miss_count++;
                result.l1_slot = select_l1_slot(state, ++age);
                if (result.l1_slot >= 0) {
                    int old_l2 = state.l1_entries[result.l1_slot].l2_slot_id;
                    if (old_l2 >= 0) state.l2_to_l1_entry.erase(old_l2);
                    state.l1_entries[result.l1_slot].l2_slot_id = result.slot;
                    state.l1_entries[result.l1_slot].age = age;

                    state.l2_to_l1_entry[result.slot] = result.l1_slot;
                    result.needs_l1_promotion = true;
                }
            }
        }
        return result;
    }

    // L2 MISS — evict
    result.slot = select_slot(state, epoch);
    if (result.slot < 0) {
        throw std::runtime_error(
            format("Flash-MoE slot overflow layer %d: need %d slots, have %d "
                   "(tokens=%lld, topk=%lld). Increase --moe-slot-bank.",
                layer, (int)((int64_t)state.resident_count + 1), state.n_slots,
                (long long)n_tokens, (long long)n_expert_used));
    }

    state.slot_reserved_epoch[result.slot] = epoch;
    result.miss = true;
    result.evicted_expert = state.slot_to_expert[result.slot];
    result.cold = (result.evicted_expert < 0);

    if (state.two_level_enabled) {
        result.l1_slot = select_l1_slot(state, ++age);
        if (result.l1_slot >= 0) {
            int old_l2 = state.l1_entries[result.l1_slot].l2_slot_id;
            if (old_l2 >= 0) state.l2_to_l1_entry.erase(old_l2);
            state.l1_entries[result.l1_slot].l2_slot_id = result.slot;
            state.l1_entries[result.l1_slot].age = age;
                    state.l2_to_l1_entry[result.slot] = result.l1_slot;
            result.needs_l1_promotion = true;
        }
    }

    return result;
}

void llama_flash_moe_slot_runtime::commit_reserved_slot(
        flash_moe_layer_state & state, int32_t expert, const reserved_slot & reserved) {
    // L2 commit
    if (reserved.miss && reserved.slot >= 0) {
        state.cache_miss_count++;
        if (reserved.evicted_expert >= 0 && reserved.evicted_expert < expert_count) {
            state.expert_to_slot[reserved.evicted_expert] = -1;
        } else {
            state.resident_count++;
            state.peak_resident_count = std::max(state.peak_resident_count, state.resident_count);
        }
        state.slot_to_expert[reserved.slot] = expert;
        state.expert_to_slot[expert] = reserved.slot;
    }

    // L1 commit handled in reserve_expert_slot via hash map
}

// ---- Epoch Wraparound ----

uint32_t llama_flash_moe_slot_runtime::next_request_epoch() {
    ++request_epoch;
    if (request_epoch != 0) return request_epoch;
    request_epoch = 1;
    for (auto & state : layers) {
        std::fill(state.slot_reserved_epoch.begin(), state.slot_reserved_epoch.end(), 0);
        std::fill(state.request_seen_epoch.begin(), state.request_seen_epoch.end(), 0);
    }
    return request_epoch;
}

// ---- Slot Selection (two-pass: free first, then LRU) ----

int32_t llama_flash_moe_slot_runtime::select_slot(
        const flash_moe_layer_state & state, uint32_t epoch) {
    // Pass 1: find a free slot
    for (int32_t slot = 0; slot < state.n_slots; ++slot) {
        if (state.slot_reserved_epoch[slot] != epoch
            && state.slot_to_expert[slot] < 0) {
            return slot;
        }
    }
    // Pass 2: LRU victim
    int32_t  victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int32_t slot = 0; slot < state.n_slots; ++slot) {
        if (state.slot_reserved_epoch[slot] == epoch) continue;
        if (state.slot_age[slot] < oldest) {
            oldest = state.slot_age[slot];
            victim = slot;
        }
    }
    return victim;
}

int32_t llama_flash_moe_slot_runtime::select_l1_slot(
        const flash_moe_layer_state & state, uint64_t) {
    for (int32_t i = 0; i < state.l1_n_slots; ++i) {
        if (state.l1_entries[i].l2_slot_id < 0) return i;
    }
    int32_t  victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int32_t i = 0; i < state.l1_n_slots; ++i) {
        if (state.l1_entries[i].age < oldest) {
            oldest = state.l1_entries[i].age;
            victim = i;
        }
    }
    if (victim >= 0) return victim;
    for (int32_t i = 0; i < state.l1_n_slots; ++i) {
        if (state.l1_entries[i].age < oldest) {
            oldest = state.l1_entries[i].age;
            victim = i;
        }
    }
    return victim;
}

// ---- I/O Engine (stub) ----

int llama_flash_moe_slot_runtime::fd_for(const std::string & repacked_path) {
    std::lock_guard<std::mutex> lock(fds_mutex);
    auto it = fds.find(repacked_path);
    if (it != fds.end()) return it->second;
#ifdef _WIN32
    int fd = _open(repacked_path.c_str(), _O_RDONLY | _O_BINARY);
#else
    int fd = open(repacked_path.c_str(), O_RDONLY);
#endif
    if (fd < 0) {
        throw std::runtime_error(
            format("Flash-MoE: cannot open sidecar file '%s'", repacked_path.c_str()));
    }
    fds[repacked_path] = fd;
    return fd;
}

void llama_flash_moe_slot_runtime::install_loads(
        flash_moe_layer_state & state,
        const std::vector<pending_slot_load> & pending_loads,
        int32_t cache_io_split,
        const std::array<int32_t, 3> & stripe) {

    // Compute effective stripe weights combined with distribute ratios
    // (element-wise product: effective[i] = stripe[i] * distribute[i])
    std::array<int32_t, 3> effective_stripe = stripe;
    {
        // Identify which stripe set was passed by value comparison
        const bool is_prefetch = (stripe[0] == prefetch_stripe_[0] && stripe[1] == prefetch_stripe_[1] && stripe[2] == prefetch_stripe_[2]);
        const bool is_prefill  = (stripe[0] == prefill_stripe_[0]  && stripe[1] == prefill_stripe_[1]  && stripe[2] == prefill_stripe_[2]);
        const auto & dist = is_prefetch ? prefetch_distribute_ : (is_prefill ? prefill_distribute_ : demand_distribute_);
        if (dist[0] != 1 || dist[1] != 1 || dist[2] != 1) {
            for (int i = 0; i < 3; ++i) {
                effective_stripe[i] = stripe[i] * dist[i];
            }
        }
    }

    // Distribute pending loads across lanes according to effective stripe ratio
    // Uses proportional assignment via cumulative thresholds for even small N.
    std::vector<pending_slot_load> loads;
    if (!pending_loads.empty()) {
        loads.reserve(pending_loads.size());
        const int32_t total_w = effective_stripe[0] + effective_stripe[1] + effective_stripe[2];
        if (total_w > 0 && (effective_stripe[1] > 0 || effective_stripe[2] > 0)) {
            const size_t n_total = pending_loads.size();
            const int32_t cum_pri = effective_stripe[0];
            const int32_t cum_sec = effective_stripe[0] + effective_stripe[1];
            for (size_t i = 0; i < n_total; ++i) {
                auto load = pending_loads[i];
                const int32_t pos = (int32_t)(i * total_w / n_total);
                load.lane = (pos < cum_pri) ? source_lane::primary
                          : (pos < cum_sec) ? source_lane::secondary
                          : source_lane::tertiary;
                loads.push_back(load);
            }
        } else {
            loads = pending_loads;
        }
    }

    // If verify sidecar is enabled, stat the sidecar files before reading
    if (moe_verify_sidecar_ && !loads.empty()) {
        std::set<std::string> verified_paths;
        for (const auto & load : loads) {
            (void)load;
            const auto * entry = state.gate_up_entry;
            if (!entry && state.gate_entry) entry = state.gate_entry;
            if (!entry && state.up_entry) entry = state.up_entry;
            if (!entry) entry = state.down_entry;
            if (!entry) continue;
            if (verified_paths.count(entry->repacked_path)) continue;
            verified_paths.insert(entry->repacked_path);
            struct stat st;
            if (stat(entry->repacked_path.c_str(), &st) != 0) {
                LLAMA_LOG_WARN("%s: verify-sidecar: cannot stat '%s' (%s)\n",
                    __func__, entry->repacked_path.c_str(), strerror(errno));
            } else {
                // Compute expected size from entries: for each family, the total bytes
                // = bytes_per_expert * expert_count + repacked_offset
                (void)st;
                LLAMA_LOG_INFO("%s: verify-sidecar: '%s' size=%zu\n",
                    __func__, entry->repacked_path.c_str(), (size_t)st.st_size);
            }
        }
    }

    for (const auto & load : loads) {
        int32_t expert = load.expert;
        int32_t slot   = load.slot;

        if (slot < 0 || slot >= state.n_slots) {
            LLAMA_LOG_ERROR("install_loads: slot %d out of range [0, %d)\n", slot, state.n_slots);
            continue;
        }
        if (expert < 0 || expert >= expert_count) {
            LLAMA_LOG_ERROR("install_loads: expert %d out of range [0, %d)\n", expert, expert_count);
            continue;
        }

        // Evict previous occupant
        int32_t evicted = state.slot_to_expert[slot];
        if (evicted >= 0 && evicted < expert_count) {
            state.expert_to_slot[evicted] = -1;
        }

        // Track install metrics for this load
        install_metrics metrics;
        const int64_t t_load_start = ggml_time_us();

        // Read expert data from sidecar
        struct {
            const llama_flash_moe_sidecar_entry * entry = nullptr;
            ggml_tensor * tensor = nullptr;
        } families[4];
        int n_families = 0;

        if (state.gate_up_entry && state.gate_up_tensor) {
            families[n_families++] = { state.gate_up_entry, state.gate_up_tensor };
        }
        if (state.gate_entry && state.gate_tensor) {
            families[n_families++] = { state.gate_entry, state.gate_tensor };
        }
        if (state.up_entry && state.up_tensor) {
            families[n_families++] = { state.up_entry, state.up_tensor };
        }
        if (state.down_entry && state.down_tensor) {
            families[n_families++] = { state.down_entry, state.down_tensor };
        }

        // Choose read strategy: mixed slot buffer (single pread) or individual
        bool use_mixed = flash_moe_mixed_slot_buffer_enabled()
            && !state.mixed_slot_fields.empty();
        std::string combined_path;

        if (use_mixed) {
            // Verify all families share the same repacked_file for a single pread
            const auto & fields = state.mixed_slot_fields;
            combined_path = fields[0].entry->repacked_path;
            for (size_t fi = 1; fi < fields.size(); ++fi) {
                if (fields[fi].entry->repacked_path != combined_path) {
                    use_mixed = false;
                    break;
                }
            }
        }

        if (use_mixed) {
            // ---- Single pread: combined slot buffer across all families ----
            const auto & fields = state.mixed_slot_fields;
            const size_t total_bytes = state.mixed_slot_bytes;
            if (total_bytes == 0) continue;

            // Resolve lane-specific repacked_offset for the combined pread
            off_t lane_base_offset;
            if (load.lane != source_lane::primary && fields[0].tensor) {
                const char * tname = ggml_get_name(fields[0].tensor);
                if (tname) {
                    const auto & lane_map = (load.lane == source_lane::secondary)
                        ? model.flash_moe_secondary_sidecar_entries()
                        : model.flash_moe_tertiary_sidecar_entries();
                    auto it = lane_map.find(tname);
                    lane_base_offset = (it != lane_map.end())
                        ? (off_t)it->second.repacked_offset
                        : (off_t)fields[0].entry->repacked_offset;
                } else {
                    lane_base_offset = (off_t)fields[0].entry->repacked_offset;
                }
            } else {
                lane_base_offset = (off_t)fields[0].entry->repacked_offset;
            }
            const off_t combined_offset = (off_t)((size_t)lane_base_offset
                + (size_t)expert * total_bytes);

            std::vector<uint8_t> combined_buf(total_bytes);

            if (resident_bank_source) {
                auto it = resident_banks.find(combined_path);
                if (it == resident_banks.end()) {
                    throw std::runtime_error(
                        format("Flash-MoE: missing resident packed bank '%s'",
                            combined_path.c_str()));
                }
                const auto & bank = it->second;
                const size_t copy_off = static_cast<size_t>(combined_offset);
                if (copy_off + total_bytes > bank.size()) {
                    throw std::runtime_error(
                        format("Flash-MoE: resident packed bank '%s' too small for combined read "
                               "(offset=%zu, need %zu bytes, have %zu bytes)",
                            combined_path.c_str(), copy_off, total_bytes, bank.size()));
                }
                std::memcpy(combined_buf.data(), bank.data() + copy_off, total_bytes);
            } else {
                int fd = fd_for(combined_path);
                if (cache_io_split > 1) {
                    // Split combined pread into cache_io_split chunks
                    const size_t chunk_size  = total_bytes / (size_t)cache_io_split;
                    const size_t remainder   = total_bytes % (size_t)cache_io_split;
                    size_t chunk_offset = 0;
                    for (int split_i = 0; split_i < cache_io_split; ++split_i) {
                        const size_t this_chunk = chunk_size
                            + (split_i == cache_io_split - 1 ? remainder : 0);
                        if (this_chunk == 0) continue;
                        ssize_t n = pread(fd, combined_buf.data() + chunk_offset,
                            this_chunk, combined_offset + (off_t)chunk_offset);
                        if (n < 0 || (size_t)n != this_chunk) {
                            throw std::runtime_error(
                                format("Flash-MoE: mixed slot pread chunk %d/%d failed "
                                       "(offset=%zu, size=%zu, path='%s'): "
                                       "expected %zu, got %zd",
                                       split_i + 1, cache_io_split,
                                       (size_t)(combined_offset + (off_t)chunk_offset),
                                       this_chunk, combined_path.c_str(),
                                       this_chunk, n));
                        }
                        chunk_offset += this_chunk;
                    }
                } else {
                    ssize_t n = pread(fd, combined_buf.data(), total_bytes, combined_offset);
                    if (n < 0 || (size_t)n != total_bytes) {
                        throw std::runtime_error(
                            format("Flash-MoE: mixed slot pread failed (offset=%zu, size=%zu, path='%s'): "
                                   "expected %zu, got %zd",
                                   (size_t)combined_offset, total_bytes,
                                   combined_path.c_str(), total_bytes, n));
                    }
                }
            }

            // Copy individual family slices from combined buffer to tensors
            for (const auto & field : fields) {
                const auto * entry = field.entry;
                auto * tensor = field.tensor;

                // Resolve lane-specific sidecar entry for non-primary I/O lanes
                if (load.lane != source_lane::primary && tensor) {
                    const char * tname = ggml_get_name(tensor);
                    if (tname) {
                        const auto & lane_map = (load.lane == source_lane::secondary)
                            ? model.flash_moe_secondary_sidecar_entries()
                            : model.flash_moe_tertiary_sidecar_entries();
                        auto it = lane_map.find(tname);
                        if (it != lane_map.end()) {
                            entry = &it->second;
                        }
                    }
                }

                const size_t family_bytes = entry->bytes_per_expert;
                const size_t dest_off = (size_t)slot * family_bytes;

#ifdef LLAMA_FLASH_MOE_GPU_BANK
                const bool on_gpu = tensor->buffer
                    && !ggml_backend_buffer_is_host(tensor->buffer);
                if (on_gpu) {
                    ggml_backend_tensor_set(tensor,
                        combined_buf.data() + field.slot_offset,
                        dest_off, family_bytes);
                } else {
                    std::memcpy((char *)tensor->data + dest_off,
                        combined_buf.data() + field.slot_offset,
                        family_bytes);
                }
#else
                std::memcpy((char *)tensor->data + dest_off,
                    combined_buf.data() + field.slot_offset,
                    family_bytes);
#endif
            }
            // Accumulate per-family byte counts for trace metrics
            for (const auto & field : fields) {
                const size_t fam_bytes = field.entry->bytes_per_expert;
                switch (field.family) {
                    case 0: metrics.bytes_gate_up += fam_bytes; break;
                    case 1: metrics.bytes_gate     += fam_bytes; break;
                    case 2: metrics.bytes_up       += fam_bytes; break;
                    case 3: metrics.bytes_down     += fam_bytes; break;
                }
            }
        } else {
            // ---- Individual pread per family with read pool ----
            LLAMA_LOG_DEBUG("%s: individual expert=%d slot=%d lane=%d families=%d split=%d\n",
                __func__, expert, slot, (int)load.lane, n_families, cache_io_split);
            std::vector<pread_task> pool_tasks;

            // Track per-family state needed after reads complete
            struct family_pread_info {
                int              task_start = -1; // first pool task index (-1 = not using pool)
                int              task_count = 0;  // number of pool tasks (1 for single, >1 for split)
                ggml_tensor *    tensor = nullptr;
                size_t           expert_bytes = 0;
                size_t           slot_offset = 0;
                bool             on_gpu = false;
                std::vector<uint8_t> staging;
            };
            std::vector<family_pread_info> family_infos;
            family_infos.reserve((size_t)n_families);

            for (int f = 0; f < n_families; ++f) {
                const auto * entry = families[f].entry;
                auto * tensor = families[f].tensor;

                // Resolve lane-specific sidecar entry for non-primary I/O lanes
                if (load.lane != source_lane::primary && tensor) {
                    const char * tname = ggml_get_name(tensor);
                    if (tname) {
                        const auto & lane_map = (load.lane == source_lane::secondary)
                            ? model.flash_moe_secondary_sidecar_entries()
                            : model.flash_moe_tertiary_sidecar_entries();
                        auto it = lane_map.find(tname);
                        if (it != lane_map.end()) {
                            entry = &it->second;
                        }
                    }
                }

                size_t expert_bytes = entry->bytes_per_expert;
                if (expert_bytes == 0) continue;

                off_t offset = (off_t)(entry->repacked_offset
                    + (size_t)expert * expert_bytes);

                size_t slot_offset = (size_t)slot * expert_bytes;

                size_t tensor_nb2 = tensor ? (size_t)tensor->nb[2] : 0;
                if (tensor && expert_bytes != tensor_nb2) {
                    LLAMA_LOG_WARN("%s: OFFSET MISMATCH expert=%d slot=%d family=%s expert_bytes=%zu tensor_nb[2]=%zu diff=%zd\n",
                        __func__, expert, slot,
                        entry->tensor_family.c_str(),
                        expert_bytes, tensor_nb2,
                        (ssize_t)((ssize_t)expert_bytes - (ssize_t)tensor_nb2));
                }

#ifdef LLAMA_FLASH_MOE_GPU_BANK
                const bool on_gpu = tensor->buffer
                    && !ggml_backend_buffer_is_host(tensor->buffer);
#else
                const bool on_gpu = false;
#endif

                family_pread_info info;
                info.tensor       = tensor;
                info.expert_bytes = expert_bytes;
                info.slot_offset  = slot_offset;
                info.on_gpu       = on_gpu;

                char * dst;
                if (on_gpu) {
                    info.staging.resize(expert_bytes);
                    dst = reinterpret_cast<char *>(info.staging.data());
                } else {
                    dst = ((char *)tensor->data) + slot_offset;
                }

                if (resident_bank_source) {
                    auto it = resident_banks.find(entry->repacked_path);
                    if (it == resident_banks.end()) {
                        throw std::runtime_error(
                            format("Flash-MoE: missing resident packed bank '%s'",
                                entry->repacked_path.c_str()));
                    }
                    const auto & bank = it->second;
                    const size_t copy_offset = static_cast<size_t>(offset);
                    if (copy_offset + expert_bytes > bank.size()) {
                        throw std::runtime_error(
                            format("Flash-MoE: resident packed bank '%s' too small for tensor "
                                   "(copy_offset=%zu, need %zu bytes, have %zu bytes)",
                                entry->repacked_path.c_str(),
                                copy_offset, expert_bytes, bank.size()));
                    }
                    std::memcpy(dst, bank.data() + copy_offset, expert_bytes);
                } else {
                    int fd = fd_for(entry->repacked_path);
                    info.task_start = (int)pool_tasks.size();
                    if (cache_io_split > 1) {
                        // Split per-family pread into cache_io_split chunks
                        const size_t chunk_size  = expert_bytes / (size_t)cache_io_split;
                        const size_t remainder   = expert_bytes % (size_t)cache_io_split;
                        size_t chunk_offset = 0;
                        for (int split_i = 0; split_i < cache_io_split; ++split_i) {
                            const size_t this_chunk = chunk_size
                                + (split_i == cache_io_split - 1 ? remainder : 0);
                            if (this_chunk == 0) continue;
                            pool_tasks.push_back({fd,
                                (char *)dst + chunk_offset,
                                (off_t)((size_t)offset + chunk_offset),
                                this_chunk, 0, 0});
                            chunk_offset += this_chunk;
                        }
                        info.task_count = cache_io_split;
                    } else {
                        pool_tasks.push_back({fd, dst, offset, expert_bytes, 0, 0});
                        info.task_count = 1;
                    }
                }

                family_infos.push_back(std::move(info));
            }

            // Execute all batched pread tasks
            if (!pool_tasks.empty()) {
                execute_pread_tasks(pool_tasks);
            }

            // Check results and perform GPU uploads
            for (auto & info : family_infos) {
                if (info.task_start >= 0) {
                    for (int ti = 0; ti < info.task_count; ++ti) {
                        const auto & t = pool_tasks[info.task_start + ti];
                        if (t.result < 0 || (size_t)t.result != t.size) {
                            throw std::runtime_error(
                                format("Flash-MoE: pread chunk %d/%d failed (offset=%zu, size=%zu): expected %zu, got %zd",
                                       ti + 1, info.task_count,
                                       (size_t)t.offset, t.size, t.size, t.result));
                        }
                    }
                }
#ifdef LLAMA_FLASH_MOE_GPU_BANK
                if (info.on_gpu) {
                    ggml_backend_tensor_set(info.tensor, info.staging.data(),
                        info.slot_offset, info.expert_bytes);
                }
#endif
                // Accumulate per-family byte counts for trace metrics
                if (info.tensor == state.gate_up_tensor) metrics.bytes_gate_up += info.expert_bytes;
                else if (info.tensor == state.gate_tensor) metrics.bytes_gate += info.expert_bytes;
                else if (info.tensor == state.up_tensor) metrics.bytes_up += info.expert_bytes;
                else if (info.tensor == state.down_tensor) metrics.bytes_down += info.expert_bytes;
            }

            if (moe_quant_map_ && moe_quant_map_[0]) {
                LLAMA_LOG_DEBUG("%s: using quant map '%s'\n", __func__, moe_quant_map_);
            }
        }

        // Finalize and record install metrics
        metrics.total_us = ggml_time_us() - t_load_start;
        LLAMA_LOG_DEBUG("%s: load expert=%d slot=%d lane=%d families=%d bytes=%llu time=%lld us\n",
            __func__, expert, slot, (int)load.lane, n_families,
            (unsigned long long)(metrics.bytes_gate_up + metrics.bytes_gate + metrics.bytes_up + metrics.bytes_down),
            (long long)metrics.total_us);
        accumulate_install_breakdown(metrics);

        // Update mappings
        state.slot_to_expert[slot] = expert;
        state.expert_to_slot[expert] = slot;
    }

#ifdef LLAMA_FLASH_MOE_GPU_BANK
    // L1 promotion: dequant L2 slot data to F32, requant to Q8_0, upload to L1 GPU tensors
    if (state.two_level_enabled && !loads.empty()) {
        for (const auto & load : loads) {
            if (!load.needs_l1_promotion || load.l1_slot < 0) continue;

            struct prom_family {
                ggml_tensor * l2;       // source L2 tensor (or first L2 for fused)
                ggml_tensor * l2_extra; // second L2 tensor for separate gate+up (nullptr if fused)
                ggml_tensor * l1;       // destination L1 tensor
            };
            std::vector<prom_family> families;

            if (state.l1_gate_up_tensor) {
                if (state.gate_up_tensor) {
                    families.push_back({state.gate_up_tensor, nullptr, state.l1_gate_up_tensor});
                } else if (state.gate_tensor && state.up_tensor) {
                    families.push_back({state.gate_tensor, state.up_tensor, state.l1_gate_up_tensor});
                }
            }
            if (state.l1_down_tensor && state.down_tensor) {
                families.push_back({state.down_tensor, nullptr, state.l1_down_tensor});
            }

            for (const auto & pf : families) {
                if (!pf.l2 || !pf.l1 || !pf.l2->data || !pf.l1->data) continue;

                const ggml_type l2_type = pf.l2->type;
                const int64_t ne0 = pf.l2->ne[0];
                const int64_t ne1 = pf.l2->ne[1];
                const int64_t n_rows = pf.l2_extra ? ne1 * 2 : ne1;
                const int64_t n_elems = ne0 * n_rows;

                size_t l1_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, ne0);
                size_t l1_slot_off = load.l1_slot >= 0 ? (size_t)load.l1_slot * l1_row_bytes * n_rows : 0;
                size_t q8_bytes = l1_row_bytes * n_rows;

                if (l2_type == GGML_TYPE_Q8_0 && !pf.l2_extra) {
                    size_t l2_slot_off = (size_t)load.slot * ggml_row_size(GGML_TYPE_Q8_0, ne0) * ne1;
                    ggml_backend_tensor_set(pf.l1, (char *)pf.l2->data + l2_slot_off,
                        l1_slot_off, q8_bytes);
                } else if (pf.l2_extra) {
                    size_t l2_row_bytes = ggml_row_size(l2_type, ne0);
                    size_t l2x_row_bytes = ggml_row_size(pf.l2_extra->type, ne0);
                    size_t l2_slot_off  = (size_t)load.slot * l2_row_bytes * ne1;
                    size_t l2x_slot_off = (size_t)load.slot * l2x_row_bytes * ne1;
                    size_t gate_bytes = l2_row_bytes * ne1;
                    size_t up_bytes   = l2x_row_bytes * ne1;
                    size_t total_bytes = gate_bytes + up_bytes;

                    std::vector<uint8_t> combined(total_bytes);
                    memcpy(combined.data(), (char*)pf.l2->data + l2_slot_off, gate_bytes);
                    memcpy(combined.data() + gate_bytes, (char*)pf.l2_extra->data + l2x_slot_off, up_bytes);

                    llama_moe_l1_promote_to_q8_0(
                        combined.data(), pf.l1, l1_slot_off, n_elems, l2_type);
                } else {
                    size_t l2_row_bytes = ggml_row_size(l2_type, ne0);
                    size_t l2_slot_off  = (size_t)load.slot * l2_row_bytes * ne1;

                    llama_moe_l1_promote_to_q8_0(
                        (char*)pf.l2->data + l2_slot_off, pf.l1, l1_slot_off, n_elems, l2_type);
                }

                if (llama_flash_moe_deep_log_enabled()) {
                    block_q8_0 result, expected;
                    ggml_backend_tensor_get(pf.l1, &result, l1_slot_off, sizeof(block_q8_0));
                    float f32_buf[256];
                    const void * l2_src = (char*)pf.l2->data + (size_t)load.slot * ggml_row_size(l2_type, ne0) * ne1;
                    if (l2_type == GGML_TYPE_Q4_K) {
                        dequantize_row_q4_K((const block_q4_K *)l2_src, f32_buf, 256);
                    } else if (l2_type == GGML_TYPE_Q6_K) {
                        dequantize_row_q6_K((const block_q6_K *)l2_src, f32_buf, 256);
                    }
                    quantize_row_q8_0_ref(f32_buf, &expected, 32);
                    if (memcmp(&result, &expected, sizeof(block_q8_0)) != 0) {
                        LLAMA_LOG_ERROR("%s: L1 MISMATCH first element GPU=%d CPU=%d\n",
                            __func__, (int)result.qs[0], (int)expected.qs[0]);
                    }
                }

                LLAMA_LOG_DEBUG("%s: L1 promotion slot=%d l1_slot=%d %s bytes=%zu ne=[%lld %lld] (%s%s->Q8_0)\n",
                    __func__, load.slot, load.l1_slot,
                    pf.l1 == state.l1_gate_up_tensor ? "gate_up" : "down",
                    q8_bytes, (long long)ne0, (long long)n_rows,
                    pf.l2_extra ? "gate+up/" : "",
                    ggml_type_name(l2_type));
            }
        }
    }
#endif

    if (pending_loads.size() > 1) {
        LLAMA_LOG_DEBUG("%s: install_loads complete: %zu loads split=%d lanes=(%d:%d:%d)\n",
            __func__, pending_loads.size(), cache_io_split,
            stripe[0], stripe[1], stripe[2]);
    }
}

// ---- Read Pool ----

void llama_flash_moe_slot_runtime::start_read_pool() {
    if (!read_pool.workers.empty()) return;
    constexpr size_t n_workers = 4;
    read_pool.shutdown = false;
    read_pool.tasks = nullptr;
    read_pool.num_tasks = 0;
    read_pool.tasks_completed = 0;
    read_pool.generation = 0;
    read_pool.completed_generation = 0;
    read_pool.workers.reserve(n_workers);
    for (size_t idx = 0; idx < n_workers; ++idx) {
        read_pool.workers.emplace_back([this, idx]() {
            int my_gen = 0;
            while (true) {
                pread_task * tasks = nullptr;
                int num_tasks = 0;
                {
                    std::unique_lock<std::mutex> lock(read_pool.mutex);
                    read_pool.work_ready.wait(lock, [this, my_gen]() {
                        return read_pool.shutdown || read_pool.generation != my_gen;
                    });
                    if (read_pool.shutdown) return;
                    my_gen = read_pool.generation;
                    tasks = read_pool.tasks;
                    num_tasks = read_pool.num_tasks;
                }
                const int worker_count = (int)read_pool.workers.size();
                for (int ti = (int)idx; ti < num_tasks; ti += worker_count) {
                    pread_task & task = tasks[ti];
                    task.result = pread(task.fd, task.dst, task.size, task.offset);
                }
                {
                    std::lock_guard<std::mutex> lock(read_pool.mutex);
                    read_pool.tasks_completed++;
                    if (read_pool.tasks_completed == worker_count) {
                        read_pool.completed_generation = my_gen;
                        read_pool.work_done.notify_one();
                    }
                }
            }
        });
    }
}

void llama_flash_moe_slot_runtime::stop_read_pool() {
    {
        std::lock_guard<std::mutex> lock(read_pool.mutex);
        read_pool.shutdown = true;
    }
    read_pool.work_ready.notify_all();
    for (auto & w : read_pool.workers) {
        if (w.joinable()) w.join();
    }
    read_pool.workers.clear();
}

void llama_flash_moe_slot_runtime::execute_pread_tasks(std::vector<pread_task> & tasks) {
    if (tasks.empty()) return;
    if (read_pool.workers.empty()) {
        for (auto & task : tasks) {
            task.result = pread(task.fd, task.dst, task.size, task.offset);
        }
        return;
    }
    int gen = 0;
    {
        std::lock_guard<std::mutex> lock(read_pool.mutex);
        read_pool.tasks = tasks.data();
        read_pool.num_tasks = (int)tasks.size();
        read_pool.tasks_completed = 0;
        read_pool.generation++;
        gen = read_pool.generation;
    }
    read_pool.work_ready.notify_all();
    std::unique_lock<std::mutex> lock(read_pool.mutex);
    read_pool.work_done.wait(lock, [this, gen]() {
        return read_pool.completed_generation >= gen || read_pool.shutdown;
    });
}

// ---- Progress reporting ----

bool llama_flash_moe_slot_runtime::progress_get_data(
        llama_flash_moe_progress_stats & out) const {
    std::memset(&out, 0, sizeof(out));
    out.available = false;
    out.prefill_profile = transient_shared_scratch;

    uint64_t total_hits       = 0;
    uint64_t total_misses     = 0;
    uint64_t total_l1_hits    = 0;
    uint64_t total_l1_misses  = 0;
    uint64_t total_experts    = 0;

    for (const auto & state : layers) {
        if (!state.enabled) continue;
        total_hits       += state.cache_hit_count;
        total_misses     += state.cache_miss_count;
        total_l1_hits    += state.l1_cache_hit_count;
        total_l1_misses  += state.l1_cache_miss_count;
        total_experts    += (uint64_t)state.peak_resident_count;
    }

    if (total_hits + total_misses > 0) {
        out.cache_hit_pct    = (double)total_hits / (double)(total_hits + total_misses) * 100.0;
        out.cache_hit_pct_l2 = out.cache_hit_pct;
        out.unique_experts   = total_experts;
        out.miss_experts     = total_misses;
        out.bytes_loaded     = total_io_bytes_;
        if (total_io_time_us_ > 0) {
            out.reload_bw_gbps = (double)total_io_bytes_ / (double)total_io_time_us_ / 1000.0;
        }
        out.available        = true;
    }

    if (total_l1_hits + total_l1_misses > 0) {
        out.cache_hit_pct_l1 = (double)total_l1_hits / (double)(total_l1_hits + total_l1_misses) * 100.0;
    }

    return out.available;
}

// ---- Prefill path (stub) ----

ggml_tensor * llama_flash_moe_slot_runtime::build_prefill_moe_tensor(
        ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int,
        const std::function<void(ggml_tensor *, const char *, int)> &) {
    return nullptr;  // TODO: layer-major prefill path
}

// ---- Oracle replay ----

void llama_flash_moe_slot_runtime::load_oracle_trace(const std::string & path) {
    LLAMA_LOG_INFO("%s: loading oracle trace from '%s'\n", __func__, path.c_str());

    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error(
            format("Flash-MoE: cannot open oracle trace file '%s'", path.c_str()));
    }

    std::string line;
    int n_records = 0;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        try {
            auto j = nlohmann::json::parse(line);
            oracle_record rec;
            rec.layer        = j.value("layer", -1);
            rec.n_tokens     = j.value("n_tokens", 0);
            auto exp_ids     = j.value("expert_ids", std::vector<int32_t>());
            rec.experts      = exp_ids;
            rec.n_expert_used = (int32_t)exp_ids.size();

            // Compute sequential slot_ids for oracle replay
            rec.slot_ids.resize(exp_ids.size());
            for (size_t i = 0; i < exp_ids.size(); ++i) {
                rec.slot_ids[i] = (int32_t)i;
            }

            oracle_records.push_back(std::move(rec));
            n_records++;
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: skipping malformed line: %s\n", __func__, e.what());
        }
    }

    LLAMA_LOG_INFO("%s: loaded %d oracle records from '%s'\n",
        __func__, n_records, path.c_str());

    prime_oracle_trace();
}

void llama_flash_moe_slot_runtime::prime_oracle_trace() {
    // Collect unique (layer, expert) pairs across all records
    std::set<std::pair<int32_t, int32_t>> unique_pairs;
    for (const auto & rec : oracle_records) {
        for (int32_t exp : rec.experts) {
            unique_pairs.insert({rec.layer, exp});
        }
    }

    LLAMA_LOG_INFO("%s: oracle_all_hit mode: %u unique (layer,expert) pairs across %zu records\n",
        __func__, (unsigned)unique_pairs.size(), oracle_records.size());

    // Attempt to pre-reserve slots for each unique pair
    int reserved = 0;
    for (const auto & [il, expert] : unique_pairs) {
        if ((size_t)il >= layers.size() || !layers[il].enabled) continue;
        auto & state = layers[il];
        int32_t slot = state.expert_to_slot[expert];
        if (slot < 0) {
            // Find a free slot
            for (int32_t s = 0; s < state.n_slots; ++s) {
                if (state.slot_to_expert[s] < 0) {
                    state.slot_to_expert[s] = expert;
                    state.expert_to_slot[expert] = s;
                    reserved++;
                    break;
                }
            }
        }
    }

    oracle_prime_stats = reserved;
    LLAMA_LOG_INFO("%s: pre-reserved %d slots for oracle_all_hit\n", __func__, reserved);
}

bool llama_flash_moe_slot_runtime::next_oracle_record(
        int layer, int n_expert_used, int n_tokens, oracle_record & out) {
    if (oracle_cursor >= oracle_records.size()) {
        return false;
    }
    for (size_t i = oracle_cursor; i < oracle_records.size(); ++i) {
        const auto & rec = oracle_records[i];
        if (rec.layer == layer && rec.n_expert_used == n_expert_used && rec.n_tokens == n_tokens) {
            out = rec;
            oracle_cursor = i + 1;
            return true;
        }
        // Also match on just layer if expert/token count didn't match (fallback)
        if (rec.layer == layer && rec.n_expert_used == n_expert_used) {
            out = rec;
            oracle_cursor = i + 1;
            return true;
        }
    }
    return false;
}

bool llama_flash_moe_slot_runtime::oracle_record_is_resident(
        const oracle_record & rec, int n_slots) {
    if (rec.experts.empty()) return false;
    std::set<int32_t> unique(rec.experts.begin(), rec.experts.end());
    return (int)unique.size() <= n_slots;
}

std::vector<int32_t> llama_flash_moe_slot_runtime::materialize_oracle_slot_ids(
        const oracle_record & rec) {
    // Return sequential slot IDs for each expert in the record.
    // Each token uses the same expert→slot mapping.
    std::vector<int32_t> ids;
    ids.reserve((size_t)rec.n_expert_used * rec.n_tokens);
    for (int32_t t = 0; t < rec.n_tokens; ++t) {
        for (int32_t e = 0; e < rec.n_expert_used; ++e) {
            ids.push_back(e);
        }
    }
    return ids;
}

void llama_flash_moe_slot_runtime::prefetch_experts(
        int il, const std::vector<int32_t> & expert_ids,
        source_lane lane) {
    if ((size_t)il >= layers.size() || !layers[il].enabled) return;
    auto & state = layers[il];

    std::vector<pending_slot_load> pending_loads;
    for (int32_t expert : expert_ids) {
        if (expert < 0 || expert >= expert_count) continue;
        int32_t slot = state.expert_to_slot[expert];
        if (slot >= 0 && slot < state.n_slots) {
            // Already resident — skip
            continue;
        }
        // Find a free slot
        slot = -1;
        for (int32_t s = 0; s < state.n_slots; ++s) {
            if (state.slot_to_expert[s] < 0) {
                slot = s;
                break;
            }
        }
        if (slot < 0) continue; // no free slot

        state.slot_to_expert[slot] = expert;
        state.expert_to_slot[expert] = slot;
        pending_loads.push_back({expert, slot, lane, -1, false});
    }

    if (!pending_loads.empty()) {
        LLAMA_LOG_DEBUG("%s: prefetch layer=%d experts=%zu lane=%d pending=%zu\n",
            __func__, il, expert_ids.size(), (int)lane, pending_loads.size());
        install_loads(state, pending_loads, prefetch_cache_io_split_, prefetch_stripe_);
    }
}

void llama_flash_moe_slot_runtime::prime_oracle_prefetch_record(int step) {
    oracle_cursor += (size_t)step;
    if (oracle_cursor > oracle_records.size()) {
        oracle_cursor = oracle_records.size();
    }
}

void llama_flash_moe_slot_runtime::accumulate_install_breakdown(
        const install_metrics & metrics) {
    install_breakdown_log.push_back(metrics);
    total_io_bytes_ += (uint64_t)(metrics.bytes_gate_up + metrics.bytes_gate + metrics.bytes_up + metrics.bytes_down);
    total_io_time_us_ += metrics.total_us;
}

// Compute the top-N most frequent experts from remaining oracle records for a layer,
// excluding experts already resident in the slot bank.
std::vector<int32_t> llama_flash_moe_slot_runtime::compute_next_hot_experts(
        int layer, int n) const {
    if (oracle_records.empty() || n <= 0) {
        return {};
    }
    std::unordered_map<int32_t, int32_t> freq;
    for (size_t i = oracle_cursor; i < oracle_records.size(); ++i) {
        const auto & rec = oracle_records[i];
        if (rec.layer != layer) continue;
        for (int32_t exp : rec.experts) {
            if (exp >= 0) freq[exp]++;
        }
    }
    if (freq.empty()) return {};
    // Sort experts by frequency descending
    std::vector<std::pair<int32_t, int32_t>> sorted(freq.begin(), freq.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });
    // Exclude already-resident experts
    std::vector<int32_t> result;
    result.reserve((size_t)n);
    if ((size_t)layer < layers.size() && layers[layer].enabled) {
        const auto & state = layers[layer];
        for (const auto & kv : sorted) {
            if (result.size() >= (size_t)n) break;
            int32_t expert = kv.first;
            if (expert >= 0 && expert < expert_count &&
                state.expert_to_slot[expert] < 0) {
                result.push_back(expert);
            }
        }
    } else {
        for (const auto & kv : sorted) {
            if (result.size() >= (size_t)n) break;
            result.push_back(kv.first);
        }
    }
    return result;
}

// ---- Backend trace ----

bool llama_flash_moe_slot_runtime::flash_moe_backend_trace_enabled() {
    const char * env = getenv("LLAMA_FLASH_MOE_BACKEND_TRACE");
    if (env == nullptr) return false;
    if (strcmp(env, "0") == 0 || strcmp(env, "false") == 0) return false;
    return env[0] != '\0';
}

int llama_flash_moe_slot_runtime::flash_moe_backend_trace_env_int(
        const char * name, int fallback) {
    const char * env = getenv(name);
    if (env == nullptr || env[0] == '\0') return fallback;
    char * end = nullptr;
    long val = strtol(env, &end, 10);
    if (end == env) return fallback;
    return (int)val;
}

bool llama_flash_moe_slot_runtime::flash_moe_log_routed_backends() {
    const char * env = getenv("LLAMA_FLASH_MOE_LOG_ROUTED_BACKENDS");
    if (env == nullptr) return false;
    if (strcmp(env, "0") == 0 || strcmp(env, "false") == 0) return false;
    return env[0] != '\0';
}

void llama_flash_moe_slot_runtime::write_trace_record(
        int layer, int n_tokens, int n_experts,
        const int32_t * expert_ids, const int32_t * slot_ids,
        uint64_t elapsed_us) {
    if (trace_fp == nullptr) return;

    std::string expert_str;
    std::string slot_str;
    for (int i = 0; i < n_experts; ++i) {
        if (i > 0) {
            expert_str += ",";
            slot_str   += ",";
        }
        expert_str += std::to_string(expert_ids[i]);
        slot_str   += std::to_string(slot_ids[i]);
    }

    fprintf(trace_fp,
        "{\"layer\":%d,\"n_tokens\":%d,\"n_experts\":%d,"
        "\"expert_ids\":[%s],\"slot_ids\":[%s],\"elapsed_us\":%llu}\n",
        layer, n_tokens, n_experts,
        expert_str.c_str(), slot_str.c_str(),
        (unsigned long long)elapsed_us);
    fflush(trace_fp);
}

void llama_flash_moe_slot_runtime::write_trace(const std::string & label) {
    if (trace_fp == nullptr) return;
    fprintf(trace_fp, "{\"label\":\"%s\"}\n", label.c_str());
    fflush(trace_fp);
}

std::string llama_flash_moe_slot_runtime::flash_moe_prepare_trace_output_path() {
    const char * env = getenv("LLAMA_FLASH_MOE_TRACE_OUTPUT");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    return "flash_moe_trace.jsonl";
}

FILE * llama_flash_moe_slot_runtime::flash_moe_open_trace_output_file(
        const std::string & path) {
    FILE * fp = fopen(path.c_str(), "w");
    if (fp == nullptr) {
        LLAMA_LOG_WARN("%s: failed to open trace output '%s'\n", __func__, path.c_str());
        return nullptr;
    }
    return fp;
}

bool llama_flash_moe_slot_runtime::flash_moe_log_colors_enabled() {
    const char * env = getenv("LLAMA_FLASH_MOE_LOG_COLORS");
    return env != nullptr && env[0] != '\0';
}

// ---- Resident-bank preloading ----

void llama_flash_moe_slot_runtime::preload_resident_banks() {
    if (!resident_bank_source) {
        LLAMA_LOG_WARN("%s: resident bank source disabled, skipping preload\n", __func__);
        return;
    }

    // Collect unique sidecar file paths referenced by all enabled layers
    std::vector<std::string> unique_paths;
    unique_paths.reserve(layers.size() * 4);
    for (const auto & state : layers) {
        if (!state.enabled) {
            continue;
        }
        for (const auto * entry : { state.gate_up_entry, state.gate_entry, state.up_entry, state.down_entry }) {
            if (entry == nullptr) {
                continue;
            }
            if (resident_banks.find(entry->repacked_path) == resident_banks.end()) {
                unique_paths.push_back(entry->repacked_path);
                resident_banks[entry->repacked_path] = std::vector<uint8_t>();
            }
        }
    }

    // Read each unique sidecar file into memory
    size_t total_bytes = 0;
    for (const auto & path : unique_paths) {
        auto & data = resident_banks[path];
        std::ifstream fin(path, std::ios::binary | std::ios::ate);
        if (!fin.is_open()) {
            throw std::runtime_error(
                format("Flash-MoE: cannot open sidecar file '%s' for resident preload", path.c_str()));
        }
        const size_t size = static_cast<size_t>(fin.tellg());
        if (size == 0) {
            throw std::runtime_error(
                format("Flash-MoE: sidecar file '%s' is empty", path.c_str()));
        }
        data.resize(size);
        fin.seekg(0, std::ios::beg);
        if (!fin.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size))) {
            throw std::runtime_error(
                format("Flash-MoE: failed to read sidecar file '%s' (%zu bytes)", path.c_str(), size));
        }
        total_bytes += size;
    }

    LLAMA_LOG_INFO("%s: preloaded %zu files (%.2f GiB)\n",
        __func__, unique_paths.size(), (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
}
