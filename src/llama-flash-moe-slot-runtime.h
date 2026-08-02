#pragma once

#include "llama.h"
#include "llama-model.h"
#include "llama-cparams.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "ggml-cpp.h"

struct ggml_tensor;
struct ggml_context;
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

// ---- Flash-MoE slot-bank runtime ----

// Abstract interface used by the graph builder (build_moe_ffn) and eval callback
struct llm_flash_moe_slot_runtime_i {
    // Eval callback path
    virtual bool wants_tensor(const ggml_tensor * t) const = 0;
    virtual bool handle_tensor(ggml_tensor * t) = 0;

    // Native slot-map path
    virtual bool uses_layer(int il) const = 0;
    virtual bool uses_native_slot_map(int layer) const = 0;
    virtual bool uses_dedicated_prefill_moe(int layer) const = 0;
    virtual ggml_tensor * build_slot_ids_tensor(
        ggml_context * ctx, ggml_tensor * selected_experts, int il) = 0;
    virtual ggml_tensor * select_routed_weight_tensor(
        int layer, ggml_tensor * tensor) = 0;
    virtual void bind_slot_ids_input(int layer, ggml_tensor * slot_ids) = 0;

    // Prefill path
    virtual ggml_tensor * build_prefill_moe_tensor(
        ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * selected_experts, ggml_tensor * weights, int layer,
        const std::function<void(ggml_tensor *, const char *, int)> & cb = {}) = 0;

    // Progress
    virtual bool progress_get_data(
        struct llama_flash_moe_progress_stats & out) const = 0;

    // Slot count (used by graph_params)
    virtual int n_slots() const = 0;

    // Prefill batch progress (called from C API)
    virtual void set_prefill_batch_progress(
        uint32_t current_batch, uint32_t total_batches,
        uint32_t batch_tokens, uint32_t total_tokens,
        uint32_t sub_batch_index, uint32_t sub_batch_total,
        uint32_t tokens_before_batch) {}

    virtual void set_moe_force_prefill_batch(bool /*v*/) {}

    virtual ~llm_flash_moe_slot_runtime_i() = default;
};

// Sidecar family identification
enum class routed_family : uint8_t {
    gate_up,
    gate,
    up,
    down,
};

// Source lane for I/O distribution
enum class source_lane : uint8_t {
    primary   = 0,
    secondary = 1,
    tertiary  = 2,
};

// Per-layer slot-bank state
struct flash_moe_layer_state {
    bool     enabled = false;
    int32_t  n_slots = 0;
    ggml_tensor * slot_ids_input = nullptr;  // inp_moe_slot_ids tensor for eval callback

    // Slot-shaped host tensors (point to virtualized model tensors)
    ggml_tensor * gate_up_tensor = nullptr;
    ggml_tensor * gate_tensor    = nullptr;
    ggml_tensor * up_tensor      = nullptr;
    ggml_tensor * down_tensor    = nullptr;

    // L1 GPU slot-bank tensors (two-level mode only)
    ggml_tensor * l1_gate_up_tensor = nullptr;
    ggml_tensor * l1_down_tensor    = nullptr;

    // Sidecar entry pointers (primary and secondary lanes)
    const struct llama_flash_moe_sidecar_entry * gate_up_entry = nullptr;
    const struct llama_flash_moe_sidecar_entry * gate_entry    = nullptr;
    const struct llama_flash_moe_sidecar_entry * up_entry      = nullptr;
    const struct llama_flash_moe_sidecar_entry * down_entry    = nullptr;

    // Slot ↔ Expert mappings (epoch-based LRU)
    std::vector<int32_t>  slot_to_expert;        // L2 slot→expert_id  (-1 = free)
    std::vector<int32_t>  expert_to_slot;        // expert_id→L2 slot (-1 = not resident)
    std::vector<uint64_t> slot_age;              // L2 LRU age per slot
    std::vector<uint32_t> slot_reserved_epoch;   // L2 reservation epoch
    std::vector<uint32_t> request_seen_epoch;    // per-expert request epoch
    std::vector<int32_t>  request_slot;          // per-expert request slot

    int32_t l1_n_slots = 0;
    struct l1_entry {
        int32_t  l2_slot_id = -1;
        uint64_t age = 0;
    };
    std::vector<l1_entry> l1_entries;
    std::unordered_map<int32_t, int> l2_to_l1_entry;
    bool     two_level_enabled = false;
    int32_t  l1_device_id = 0;   // CUDA device for L1 GPU tensors (0 = default)

    // Pending I/O
    std::vector<int32_t> pending_slot_loads;     // slots needing install_loads()

    // Prediction state
    std::vector<int32_t> predicted_experts;
    std::vector<int32_t> current_token_experts;
    std::vector<int32_t> temporal_prefetch_experts;
    bool     temporal_prefetch_active = false;

    // Mixed-slot fields for multi-lane I/O
    struct mixed_slot_field {
        ggml_tensor * tensor = nullptr;
        const struct llama_flash_moe_sidecar_entry * entry = nullptr;
        int family = 0; // 0=gate_up,1=gate,2=up,3=down
        size_t slot_offset = 0;
    };
    std::vector<mixed_slot_field> mixed_slot_fields;
    std::vector<mixed_slot_field> mixed_prefetch_slot_fields;
    std::vector<mixed_slot_field> mixed_secondary_slot_fields;
    std::vector<mixed_slot_field> mixed_tertiary_slot_fields;
    size_t mixed_slot_bytes = 0;
    size_t mixed_prefetch_slot_bytes = 0;
    size_t mixed_secondary_slot_bytes = 0;
    size_t mixed_tertiary_slot_bytes = 0;

         // Stats
         int32_t  resident_count = 0;
         int32_t  peak_resident_count = 0;
         uint64_t cache_miss_count = 0;
         uint64_t cache_hit_count  = 0;
         uint64_t l1_cache_hit_count  = 0;
         uint64_t l1_cache_miss_count = 0;

         // Data integrity: slot_index → FNV-1a hash of gate_up tensor data
         std::unordered_map<int32_t, uint64_t> slot_data_hash;
     };

// Result of slot reservation (returned by reserve_expert_slot)
struct reserved_slot {
    int32_t slot             = -1;  // L2 slot index
    int32_t evicted_expert   = -1;  // expert evicted from L2 (-1 = cold)
    bool    miss             = false; // true = L2 miss (data must be loaded)
    bool    cold             = false; // true = slot was empty before
    int32_t l1_slot          = -1;  // L1 GPU slot (-1 = N/A)
    bool    needs_l1_promotion = false; // L2→L1 copy needed
    int32_t l1_evicted_expert = -1;  // expert evicted from L1
};

// Pending I/O load for a single slot
struct pending_slot_load {
    int32_t  expert = -1;
    int32_t  slot   = -1;
    source_lane lane = source_lane::primary;
    int32_t  l1_slot = -1;
    bool     needs_l1_promotion = false;
};

// Read thread pool for parallel pread() operations
struct pread_task {
    int     fd       = -1;
    void *  dst      = nullptr;
    off_t   offset   = 0;
    size_t  size     = 0;
    ssize_t result   = 0;
    int64_t elapsed_us = 0;
};

struct read_thread_pool {
    std::mutex               mutex;
    std::condition_variable  work_ready;
    std::condition_variable  work_done;
    std::vector<std::thread> workers;
    pread_task * tasks     = nullptr;
    int          num_tasks = 0;
    int          tasks_completed = 0;
    int          generation = 0;
    int          completed_generation = 0;
    bool         shutdown = false;
};

// ---- Oracle replay record ----
struct oracle_record {
    int32_t layer = -1;
    int32_t n_expert_used = 0;
    int32_t n_tokens = 0;
    std::vector<int32_t> experts;
    std::vector<int32_t> slot_ids;
};

// ---- Install I/O metrics (per-install) ----
struct install_metrics {
    int64_t pread_us = 0;
    int64_t gpu_upload_us = 0;
    int64_t total_us = 0;
    uint64_t bytes_gate_up = 0;
    uint64_t bytes_gate = 0;
    uint64_t bytes_up = 0;
    uint64_t bytes_down = 0;
};

// ---- Routed I/O metrics (cumulative) ----
struct routed_metrics {
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t bytes_loaded = 0;
    uint64_t l1_promotions = 0;
    uint64_t pread_count = 0;
    int64_t total_pread_us = 0;
    int64_t total_gpu_us = 0;
};

// Main Flash-MoE slot-bank runtime class
class llama_flash_moe_slot_runtime : public llm_flash_moe_slot_runtime_i {
public:
    llama_flash_moe_slot_runtime(
        const struct llama_model & model,
        bool perf_profile = false,
        bool transient_shared_scratch = false,
        bool l1_dequant_q8_0 = false);

    ~llama_flash_moe_slot_runtime() override;

    // ---- Eval callback interface ----
    bool wants_tensor(const ggml_tensor * t) const override;
    bool handle_tensor(ggml_tensor * t) override;

    // ---- Native slot-map interface ----
    bool uses_layer(int il) const override;
    bool uses_native_slot_map(int layer) const override;
    bool uses_dedicated_prefill_moe(int layer) const override;
    ggml_tensor * build_slot_ids_tensor(
        ggml_context * ctx, ggml_tensor * selected_experts, int il) override;
    ggml_tensor * select_routed_weight_tensor(
        int layer, ggml_tensor * tensor) override;
    void bind_slot_ids_input(int layer, ggml_tensor * slot_ids) override;

    // ---- Slot reservation (core LRU) ----
    reserved_slot reserve_expert_slot(
        flash_moe_layer_state & state, int layer, int32_t expert,
        uint32_t epoch, int64_t n_tokens, int64_t n_expert_used);
    void commit_reserved_slot(
        flash_moe_layer_state & state, int32_t expert, const reserved_slot & reserved);

    // ---- I/O engine ----
    void install_loads(flash_moe_layer_state & state,
        const std::vector<pending_slot_load> & pending_loads,
        int32_t cache_io_split = 1,
        const std::array<int32_t, 3> & stripe = {1, 0, 0});

    // ---- Prefill ----
    ggml_tensor * build_prefill_moe_tensor(
        ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * selected_experts, ggml_tensor * weights, int layer,
        const std::function<void(ggml_tensor *, const char *, int)> & cb = {}) override;

    // ---- Progress ----
    bool progress_get_data(
        struct llama_flash_moe_progress_stats & out) const override;

    // ---- Lifecycle ----
    int  n_slots() const override { return slot_count; }

    // ---- Resident-bank preloading ----
    void preload_resident_banks();

    // ---- Oracle replay ----
    void load_oracle_trace(const std::string & path);
    void prime_oracle_trace();
    bool next_oracle_record(int layer, int n_expert_used, int n_tokens, oracle_record & out);
    static bool oracle_record_is_resident(const oracle_record & rec, int n_slots);
    static std::vector<int32_t> materialize_oracle_slot_ids(const oracle_record & rec);
    void prefetch_experts(int il, const std::vector<int32_t> & expert_ids,
        source_lane lane = source_lane::primary);
    void prime_oracle_prefetch_record(int step);
    void accumulate_install_breakdown(const install_metrics & metrics);

    // ---- Mode control ----
    void set_moe_mode(int32_t mode) { moe_mode = mode; }
    int32_t get_moe_mode() const { return moe_mode; }

    // ---- Prediction and sorting parameters ----
    void set_moe_predict_prev_token(int32_t v) { moe_predict_prev_token_ = v; }
    void set_moe_predict_top1_prev(int32_t v) { moe_predict_top1_prev_ = v; }
    void set_moe_sort_decode_expert_ids(int32_t v) { moe_sort_decode_expert_ids_ = v; }

    // ---- I/O split parameters ----
    void set_io_split_params(int32_t cache_split, int32_t prefill_split, int32_t prefetch_split);

    // ---- Prefill I/O mode ----
    void set_prefill_io_mode(bool v) { prefill_io_mode_ = v; }

    // ---- Stripe/distribution parameters ----
    void set_stripe_params(
        const std::string & demand_stripe,
        const std::string & prefill_stripe,
        const std::string & prefetch_stripe);

    // ---- Prefill next-hot parameters ----
    void set_prefill_next_hot_params(int32_t experts, bool exclusive_drives);

    // ---- Quant map ----
    void set_moe_quant_map(const char * v);

    // ---- Temporal prefetch ----
    void set_moe_prefetch_temporal(int32_t v) { moe_prefetch_temporal_ = v; }
    void set_moe_prefetch_temporal_sparse(bool v) { moe_prefetch_temporal_sparse_ = v; }

    // ---- Sidecar verification ----
    void set_moe_verify_sidecar(bool v) { moe_verify_sidecar_ = v; }

    // ---- Force prefill batch ----
    void set_moe_force_prefill_batch(bool v) { moe_force_prefill_batch_ = v; }

    // ---- Distribute ratio parameters ----
    void set_distribute_params(
        const std::string & demand_distribute,
        const std::string & prefill_distribute,
        const std::string & prefetch_distribute);

    // ---- Trace harness ----
    void set_trace_harness(bool v);

    // ---- Layer state access ----
    const flash_moe_layer_state & get_layer_state(int il) const { return layers[il]; }

    // 9-arg constructor for prefill variant (stores timing pointers)
    llama_flash_moe_slot_runtime(
        const struct llama_model & model, bool perf_profile,
        bool transient_shared_scratch, bool l1_dequant_q8_0,
        int32_t prefill_micro_batch_tokens,
        const int64_t * prefill_eval_us, const int64_t * decode_eval_us,
        const int64_t * prefill_eval_tokens, const int64_t * decode_eval_tokens);

    // ---- Backend trace ----
    static bool flash_moe_backend_trace_enabled();
    static int flash_moe_backend_trace_env_int(const char * name, int fallback);
    static bool flash_moe_log_routed_backends();
    void write_trace_record(int layer, int n_tokens, int n_experts, const int32_t * expert_ids, const int32_t * slot_ids, uint64_t elapsed_us);
    void write_trace(const std::string & label);
    static std::string flash_moe_prepare_trace_output_path();
    FILE * flash_moe_open_trace_output_file(const std::string & path);
    static bool flash_moe_log_colors_enabled();

private:
    const struct llama_model & model;
    int32_t slot_count = 0;
    int32_t l1_slot_count = 0;
    int32_t expert_count = 0;
    bool    two_level_enabled = false;
    bool    transient_shared_scratch = false;
    bool    resident_bank_source = false;

    // Prediction and sorting parameters
    int32_t moe_predict_prev_token_ = 0;
    int32_t moe_predict_top1_prev_ = 0;
    int32_t moe_sort_decode_expert_ids_ = 0;

    // I/O split parameters
    int32_t cache_io_split_ = 1;            // primary decode lane
    int32_t prefill_cache_io_split_ = 1;    // prefill lane (0 = follow cache_io_split_)
    int32_t prefetch_cache_io_split_ = 1;   // prefetch lane (0 = follow cache_io_split_)

    // Prefill I/O mode flag
    bool prefill_io_mode_ = false;

    // Stripe ratios for I/O lane distribution (primary:secondary:tertiary)
    std::array<int32_t, 3> demand_stripe_  = {1, 0, 0};
    std::array<int32_t, 3> prefill_stripe_ = {1, 0, 0};
    std::array<int32_t, 3> prefetch_stripe_ = {1, 0, 0};

    // Distribute ratios (lane multiplier applied alongside stripe)
    std::array<int32_t, 3> demand_distribute_    = {1, 1, 1};
    std::array<int32_t, 3> prefill_distribute_   = {1, 1, 1};
    std::array<int32_t, 3> prefetch_distribute_  = {1, 1, 1};

    // Quant map path
    const char* moe_quant_map_ = nullptr;

    // Temporal prefetch control
    int32_t moe_prefetch_temporal_          = 0;
    bool    moe_prefetch_temporal_sparse_   = false;

    // Sidecar verification
    bool moe_verify_sidecar_ = false;

    // Force prefill batch
    bool moe_force_prefill_batch_ = false;

    // Prefill next-hot experts params
    int32_t prefill_next_hot_experts_ = 0;
    bool    prefill_next_hot_exclusive_drives_ = false;

    // Oracle trace harness flag
    bool trace_harness_mode_ = false;

    std::vector<flash_moe_layer_state> layers;

    // L1 GPU tensor allocation (keeps ggml_context and backend buffer alive)
    std::vector<ggml_context_ptr>       l1_ctxs;
    std::vector<ggml_backend_buffer_ptr> l1_bufs;

    // L2 slot bank tensor buffers (freed in destructor)
    std::vector<ggml_backend_buffer_ptr> l2_buffers;

    std::atomic<uint64_t> age{0};
    uint32_t request_epoch = 1;
    uint32_t next_request_epoch();

    // Slot selection
    static int32_t select_slot(const flash_moe_layer_state & state, uint32_t epoch);
    static int32_t select_l1_slot(const flash_moe_layer_state & state, uint64_t age);

    // Tensor name parsing
    static bool parse_topk_layer(const char * name, int & layer);

    // Tensor binding
    void bind_tensor(ggml_tensor * tensor, ggml_tensor *& tensor_out,
        const llama_flash_moe_sidecar_entry *& entry_out, bool & enabled);

    // Read pool
    read_thread_pool read_pool;
    void start_read_pool();
    void stop_read_pool();
    void execute_pread_tasks(std::vector<pread_task> & tasks);

    // Sidecar file descriptors (lazy-open, keyed by repacked_path)
    std::map<std::string, int> fds;
    std::mutex fds_mutex;
    int fd_for(const std::string & repacked_path);

    // Resident-bank preloaded data (keyed by sidecar repacked_path)
    std::unordered_map<std::string, std::vector<uint8_t>> resident_banks;

    // Oracle replay state
    std::vector<oracle_record> oracle_records;
    size_t oracle_cursor = 0;
    int oracle_prime_stats = 0;
    int oracle_prefetch_repairs = 0;
    int32_t moe_mode = 0;

    // Compute next-hot experts from remaining oracle records
    std::vector<int32_t> compute_next_hot_experts(int layer, int n) const;

    mutable uint64_t total_io_bytes_  = 0;
    mutable int64_t  total_io_time_us_ = 0;

    // Install metrics breakdown log
    std::vector<install_metrics> install_breakdown_log;

    // L1 dequant Q8_0 flag
    bool l1_dequant_q8_0_ = false;

    // Backend trace file
    FILE * trace_fp = nullptr;

    // ---- Per-decode-step stats accumulator ----
public:
    struct decode_step_stats {
        int64_t  step_number         = 0;
        int32_t  n_layers            = 0;
        int32_t  n_layers_processed  = 0;
        int32_t  total_experts       = 0;
        int64_t  n_tokens            = 0;
        int32_t  n_io_loads          = 0;
        size_t   io_bytes            = 0;
        int32_t  n_l1_promotions     = 0;
        int32_t  n_cache_hits        = 0;
        int32_t  n_cuda_replay_ok    = 0;
        int32_t  n_cuda_replay_fail  = 0;
        int32_t  n_oracle_bypass     = 0;
        int32_t  n_slot_range_errors = 0;
        int32_t  n_expert_range_errors = 0;
        int32_t  n_hash_mismatches   = 0;
        int32_t  slot_capacity       = 0;
        int32_t  expert_count_v      = 0;
        // Representative mapping (first 8 expert→slot from layer 0)
        std::vector<int32_t> rep_expert_ids;
        std::vector<int32_t> rep_slot_ids;
    };
    decode_step_stats step_stats_;

};
