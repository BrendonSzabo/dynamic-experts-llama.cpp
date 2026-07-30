#include "llama-dyn-ex.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <cerrno>
#include <cstdio>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// dtype codes from convert-gguf-to-expert-binary.py (GGML_TYPE_TO_DTYPE_CODE)
static ggml_type dyn_ex_code_to_ggml_type(uint8_t code) {
    switch (code) {
        case 0:  return GGML_TYPE_F16;
        case 1:  return GGML_TYPE_BF16;
        case 2:  return GGML_TYPE_F32;
        case 20: return GGML_TYPE_Q4_0;
        case 21: return GGML_TYPE_Q4_1;
        case 22: return GGML_TYPE_Q5_0;
        case 23: return GGML_TYPE_Q5_1;
        case 24: return GGML_TYPE_Q8_0;
        case 25: return GGML_TYPE_Q8_1;
        case 30: return GGML_TYPE_Q2_K;
        case 31: return GGML_TYPE_Q3_K;
        case 32: return GGML_TYPE_Q4_K;
        case 33: return GGML_TYPE_Q5_K;
        case 34: return GGML_TYPE_Q6_K;
        case 40: return GGML_TYPE_IQ4_NL;
        default: return GGML_TYPE_F32;
    }
}

static size_t dyn_ex_ggml_type_size(ggml_type type) {
    return ggml_type_size(type) / ggml_blck_size(type);
}

static inline uint32_t read_u32_le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t read_u64_le(const uint8_t * p) {
    return (uint64_t)read_u32_le(p) | ((uint64_t)read_u32_le(p + 4) << 32);
}

static inline uint16_t read_u16_le(const uint8_t * p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static size_t round_up_page(size_t n) {
    const size_t page = 4096;
    return ((n + page - 1) / page) * page;
}

dyn_ex_reader * dyn_ex_reader_open(const char * path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LLAMA_LOG_ERROR("dyn-ex: failed to open %s: %s\n", path, strerror(errno));
        return nullptr;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        LLAMA_LOG_ERROR("dyn-ex: fstat failed: %s\n", strerror(errno));
        close(fd);
        return nullptr;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < DYN_EX_HEADER_SIZE) {
        LLAMA_LOG_ERROR("dyn-ex: file too small (%zu bytes)\n", file_size);
        close(fd);
        return nullptr;
    }

    void * addr = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        LLAMA_LOG_ERROR("dyn-ex: mmap failed: %s\n", strerror(errno));
        close(fd);
        return nullptr;
    }

    const uint8_t * hdr = (const uint8_t *)addr;

    // validate magic
    if (memcmp(hdr, DYN_EX_MAGIC, 6) != 0) {
        LLAMA_LOG_ERROR("dyn-ex: bad magic: %.6s\n", hdr);
        munmap(addr, file_size);
        close(fd);
        return nullptr;
    }

    uint32_t version = read_u32_le(hdr + 6);
    if (version != DYN_EX_VERSION) {
        LLAMA_LOG_ERROR("dyn-ex: unsupported version %u (expected %d)\n", version, DYN_EX_VERSION);
        munmap(addr, file_size);
        close(fd);
        return nullptr;
    }

    dyn_ex_reader * r = new dyn_ex_reader();
    r->fd         = fd;
    r->mmap_addr  = addr;
    r->mmap_size  = file_size;
    r->n_layers   = (int)read_u32_le(hdr + 10);
    r->n_experts  = (int)read_u32_le(hdr + 14);
    r->expert_stride = (int64_t)read_u64_le(hdr + 18);
    r->n_params   = (int)read_u16_le(hdr + 26);

    if (r->n_params > 8) {
        LLAMA_LOG_ERROR("dyn-ex: too many params (%d, max 8)\n", r->n_params);
        delete r;
        munmap(addr, file_size);
        close(fd);
        return nullptr;
    }

    // parse param specs
    size_t off = 28;
    for (int i = 0; i < r->n_params; i++) {
        uint8_t name_len = hdr[off]; off++;
        if (off + name_len > DYN_EX_HEADER_SIZE) {
            LLAMA_LOG_ERROR("dyn-ex: header parse error at param %d\n", i);
            delete r;
            munmap(addr, file_size);
            close(fd);
            return nullptr;
        }
        memcpy(r->params[i].name, hdr + off, name_len);
        r->params[i].name[name_len] = '\0';
        off += name_len;

        uint8_t ndim = hdr[off]; off++;
        r->params[i].ndim = ndim;
        for (int d = 0; d < ndim && d < 4; d++) {
            r->params[i].shape[d] = (int64_t)read_u32_le(hdr + off);
            off += 4;
        }
        for (int d = ndim; d < 4; d++) {
            r->params[i].shape[d] = 1;
        }

        uint8_t dtype_code = hdr[off]; off++;
        r->params[i].dtype_code = dtype_code;
        r->params[i].type        = dyn_ex_code_to_ggml_type(dtype_code);
    }

    // compute per-param data offsets and strides
    // layout: each param's data is contiguous for all layers+experts
    // [param0: L*E blocks] [param1: L*E blocks] ...
    size_t data_off = DYN_EX_HEADER_SIZE;
    for (int i = 0; i < r->n_params; i++) {
        int64_t numel = 1;
        for (int d = 0; d < r->params[i].ndim; d++) {
            numel *= r->params[i].shape[d];
        }
        size_t per_expert_bytes = (size_t)numel * dyn_ex_ggml_type_size(r->params[i].type);
        size_t stride           = round_up_page(per_expert_bytes);

        r->param_data_off[i] = data_off;
        r->param_stride[i]   = stride;

        data_off += (size_t)r->n_layers * (size_t)r->n_experts * stride;
    }

    LLAMA_LOG_INFO("dyn-ex: opened %s: %d layers, %d experts, %d params\n",
                   path, r->n_layers, r->n_experts, r->n_params);

    return r;
}

void dyn_ex_reader_close(dyn_ex_reader * r) {
    if (!r) return;
    munmap(r->mmap_addr, r->mmap_size);
    close(r->fd);
    delete r;
}

size_t dyn_ex_read_param(const dyn_ex_reader * r, int param_idx, int layer, int expert_id,
                         void * buf, size_t buf_size) {
    if (param_idx < 0 || param_idx >= r->n_params) return 0;
    if (layer < 0 || layer >= r->n_layers)       return 0;
    if (expert_id < 0 || expert_id >= r->n_experts) return 0;

    size_t expert_size = dyn_ex_param_size(r, param_idx);
    if (buf_size < expert_size) return 0;

    // offset = param_data_off + (layer * n_experts + expert_id) * param_stride
    size_t file_off = r->param_data_off[param_idx]
                    + (size_t)(layer * r->n_experts + expert_id) * r->param_stride[param_idx];

    if (file_off + expert_size > r->mmap_size) return 0;

    memcpy(buf, (const uint8_t *)r->mmap_addr + file_off, expert_size);
    return expert_size;
}

size_t dyn_ex_param_size(const dyn_ex_reader * r, int param_idx) {
    if (param_idx < 0 || param_idx >= r->n_params) return 0;
    int64_t numel = 1;
    for (int d = 0; d < r->params[param_idx].ndim; d++) {
        numel *= r->params[param_idx].shape[d];
    }
    ggml_type type = r->params[param_idx].type;
    return (size_t)numel * ggml_type_size(type) / ggml_blck_size(type);
}

int dyn_ex_param_index(const dyn_ex_reader * r, const char * name) {
    for (int i = 0; i < r->n_params; i++) {
        if (strcmp(r->params[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

// ── slot cache ──────────────────────────────────────────────────────────

static size_t align_up(size_t n, size_t align) {
    return ((n + align - 1) / align) * align;
}

// helper: write data to a raw backend buffer at offset via a dummy tensor
static void raw_buf_write(ggml_backend_buffer_t buf, size_t offset, const void * data, size_t size) {
    ggml_tensor dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.buffer = buf;
    dummy.data   = (char *)ggml_backend_buffer_get_base(buf) + offset;
    dummy.type   = GGML_TYPE_I8;
    dummy.ne[0]  = (int64_t)size;
    dummy.ne[1]  = 1;
    dummy.ne[2]  = 1;
    dummy.ne[3]  = 1;
    dummy.nb[0]  = 1;
    ggml_backend_tensor_set(&dummy, data, 0, size);
}

dyn_ex_cache * dyn_ex_cache_init(
    dyn_ex_reader * reader,
    int n_slots,
    ggml_backend_dev_t dev,
    ggml_tensor * /* expert_gate_up */,
    ggml_tensor * /* expert_gate */,
    ggml_tensor * /* expert_up */,
    ggml_tensor * expert_down) {

    if (!reader || n_slots < 1 || !dev) return nullptr;
    if (!expert_down) return nullptr;

    // validate n_slots is power of 2
    if ((n_slots & (n_slots - 1)) != 0) {
        LLAMA_LOG_ERROR("dyn-ex: n_slots must be power of 2, got %d\n", n_slots);
        return nullptr;
    }

    int n_layers  = reader->n_layers;
    int n_experts = reader->n_experts;

    int pi_gate_up = dyn_ex_param_index(reader, "gate_up_proj");
    int pi_gate    = dyn_ex_param_index(reader, "gate_proj");
    int pi_up      = dyn_ex_param_index(reader, "up_proj");
    int pi_down    = dyn_ex_param_index(reader, "down_proj");

    if (pi_down < 0) {
        LLAMA_LOG_ERROR("dyn-ex: down_proj param not found in .bin\n");
        return nullptr;
    }
    if (pi_gate_up < 0 && (pi_gate < 0 || pi_up < 0)) {
        LLAMA_LOG_ERROR("dyn-ex: neither gate_up_proj nor gate_proj+up_proj found in .bin\n");
        return nullptr;
    }

    dyn_ex_cache * cache = new dyn_ex_cache();
    cache->reader    = reader;
    cache->n_layers  = n_layers;
    cache->n_experts = n_experts;
    cache->n_slots   = n_slots;
    cache->pi_gate_up = pi_gate_up;
    cache->pi_gate    = pi_gate;
    cache->pi_up      = pi_up;
    cache->pi_down    = pi_down;

    // compute per-expert sizes
    if (pi_gate_up >= 0) {
        cache->gate_up_expert_size = dyn_ex_param_size(reader, pi_gate_up);
    }
    if (pi_gate >= 0) {
        cache->gate_expert_size = dyn_ex_param_size(reader, pi_gate);
    }
    if (pi_up >= 0) {
        cache->up_expert_size = dyn_ex_param_size(reader, pi_up);
    }
    cache->down_expert_size = dyn_ex_param_size(reader, pi_down);

    // init host arrays
    cache->h_slot_of.assign((size_t)n_layers * n_experts, DYN_EX_SENTINEL);
    cache->h_expert_in.assign((size_t)n_layers * n_slots, DYN_EX_SENTINEL);
    cache->h_slot_used.assign((size_t)n_layers * n_slots, 0);

    // create GPU buffers
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // slot_map buffer: [n_layers * n_expert] int32
    size_t slot_map_bytes = (size_t)n_layers * n_experts * sizeof(int32_t);
    cache->buf_slot_map.reset(ggml_backend_buft_alloc_buffer(buft, slot_map_bytes));
    if (!cache->buf_slot_map) { dyn_ex_cache_free(cache); return nullptr; }
    // GPU buffer base is a device pointer — must use tensor_set, not memset
    {
        std::vector<uint8_t> init_buf(slot_map_bytes, 0);
        raw_buf_write(cache->buf_slot_map.get(), 0, init_buf.data(), slot_map_bytes);
    }

    // gate_up slot buffer (or gate + up separately)
    cache->gate_up_stride = align_up(cache->gate_up_expert_size, 256);
    if (pi_gate_up >= 0) {
        size_t gate_up_buf_size = (size_t)n_layers * n_slots * cache->gate_up_stride;
        cache->buf_gate_up.reset(ggml_backend_buft_alloc_buffer(buft, gate_up_buf_size));
        if (!cache->buf_gate_up) { dyn_ex_cache_free(cache); return nullptr; }
    }
    if (pi_gate >= 0 && pi_up >= 0) {
        cache->gate_stride = align_up(cache->gate_expert_size, 256);
        cache->up_stride   = align_up(cache->up_expert_size, 256);
        size_t gate_buf_size = (size_t)n_layers * n_slots * cache->gate_stride;
        size_t up_buf_size   = (size_t)n_layers * n_slots * cache->up_stride;
        cache->buf_gate.reset(ggml_backend_buft_alloc_buffer(buft, gate_buf_size));
        cache->buf_up.reset(ggml_backend_buft_alloc_buffer(buft, up_buf_size));
        if (!cache->buf_gate || !cache->buf_up) { dyn_ex_cache_free(cache); return nullptr; }
    }

    // down slot buffer
    cache->down_stride = align_up(cache->down_expert_size, 256);
    {
        size_t down_buf_size = (size_t)n_layers * n_slots * cache->down_stride;
        cache->buf_down.reset(ggml_backend_buft_alloc_buffer(buft, down_buf_size));
        if (!cache->buf_down) { dyn_ex_cache_free(cache); return nullptr; }
    }

    cache->t_gate_up.resize(n_layers, nullptr);
    cache->t_gate.resize(n_layers, nullptr);
    cache->t_up.resize(n_layers, nullptr);
    cache->t_down.resize(n_layers, nullptr);

    // async prefetch infrastructure
    {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.async && props.caps.events && props.caps.host_buffer) {
            cache->copy_backend = ggml_backend_dev_init(dev, nullptr);
            if (cache->copy_backend) {
                int n_total_slots = n_layers * n_slots;
                cache->copy_events.resize(n_total_slots, nullptr);
                for (int i = 0; i < n_total_slots; i++) {
                    cache->copy_events[i] = ggml_backend_event_new(dev);
                }
                // 4 pinned staging buffers for ring-buffer async copies
                size_t max_expert = cache->gate_up_expert_size;
                if (cache->down_expert_size > max_expert) max_expert = cache->down_expert_size;
                for (int i = 0; i < 4; i++) {
                    cache->staging_bufs.emplace_back(max_expert);
                }
                LLAMA_LOG_INFO("dyn-ex: async prefetch enabled\n");
            }
        }
    }

    LLAMA_LOG_INFO("dyn-ex: cache init: %d layers, %d experts, %d slots, "
                   "gate_up=%zuB/expert, down=%zuB/expert\n",
                   n_layers, n_experts, n_slots,
                   cache->gate_up_expert_size, cache->down_expert_size);

    return cache;
}

void dyn_ex_cache_free(dyn_ex_cache * cache) {
    if (!cache) return;
    // wait for async copies before freeing
    if (cache->copy_backend) {
        for (auto ev : cache->copy_events) {
            if (ev) {
                ggml_backend_event_synchronize(ev);
                ggml_backend_event_free(ev);
            }
        }
        ggml_backend_free(cache->copy_backend);
    }
    cache->buf_gate_up.reset();
    cache->buf_gate.reset();
    cache->buf_up.reset();
    cache->buf_down.reset();
    cache->buf_slot_map.reset();
    delete cache;
}

void dyn_ex_cache_ensure(dyn_ex_cache * cache, int layer, const int * expert_ids, int n_ids) {
    if (!cache || layer < 0 || layer >= cache->n_layers) return;
    if (!expert_ids || n_ids <= 0) return;

    fprintf(stderr, "dyn-ex ensure: L%d loading %d experts: [%d,%d,%d,%d,%d,%d,%d,%d] gate_buf=%p\n",
        layer, n_ids, expert_ids[0], n_ids>1?expert_ids[1]:-1, n_ids>2?expert_ids[2]:-1, n_ids>3?expert_ids[3]:-1,
        n_ids>4?expert_ids[4]:-1, n_ids>5?expert_ids[5]:-1, n_ids>6?expert_ids[6]:-1, n_ids>7?expert_ids[7]:-1,
        (void*)(cache->buf_gate ? ggml_backend_buffer_get_base(cache->buf_gate.get()) : nullptr));

    const int n_experts = cache->n_experts;
    const int n_slots   = cache->n_slots;
    const int layer_off_expert = layer * n_experts;
    const int layer_off_slot   = layer * n_slots;

    // CPU staging buffer
    size_t max_expert_size = cache->gate_up_expert_size;
    if (cache->down_expert_size > max_expert_size) max_expert_size = cache->down_expert_size;
    std::vector<uint8_t> cpu_buf(max_expert_size);

    bool slot_map_changed = false;
    int next_slot = 0; // simple round-robin for now

    for (int i = 0; i < n_ids; i++) {
        int eid = expert_ids[i];
        if (eid < 0 || eid >= n_experts) continue;

        // already loaded?
        int existing_slot = cache->h_slot_of[layer_off_expert + eid];
        if (existing_slot != DYN_EX_SENTINEL) continue;

        // find a free slot (round-robin reuse)
        int slot = -1;
        for (int attempt = 0; attempt < n_slots; attempt++) {
            int s = (next_slot + attempt) % n_slots;
            if (!cache->h_slot_used[layer_off_slot + s]) {
                slot = s;
                break;
            }
        }
        if (slot < 0) {
            // all slots in use: evict current next_slot
            slot = next_slot;
            int victim = cache->h_expert_in[layer_off_slot + slot];
            // keep old slot_map entry pointing to this slot (stale but valid index)
            // predictor will prefetch to fix staleness; DYN_EX_SENTINEL would crash ggml_mul_mat_id
        }
        next_slot = (slot + 1) % n_slots;

        // wait for any in-flight async copy on this slot before reusing it
        if (cache->copy_backend) {
            int ev_idx = layer_off_slot + slot;
            auto & ev = cache->copy_events[ev_idx];
            if (ev) {
                ggml_backend_event_synchronize(ev);
            }
        }

        // load gate_up weights from .bin → CPU → GPU
        if (cache->buf_gate_up && cache->pi_gate_up >= 0) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_gate_up, layer, eid,
                                         cpu_buf.data(), cache->gate_up_expert_size);
            if (n == cache->gate_up_expert_size) {
                size_t slot_off = (size_t)(layer_off_slot + slot) * cache->gate_up_stride;
                raw_buf_write(cache->buf_gate_up.get(), slot_off, cpu_buf.data(), n);
            }
        }
        if (cache->buf_gate && cache->pi_gate >= 0) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_gate, layer, eid,
                                         cpu_buf.data(), cache->gate_expert_size);
            if (n == cache->gate_expert_size) {
                size_t slot_off = (size_t)(layer_off_slot + slot) * cache->gate_stride;
                raw_buf_write(cache->buf_gate.get(), slot_off, cpu_buf.data(), n);
            }
        }
        if (cache->buf_up && cache->pi_up >= 0) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_up, layer, eid,
                                         cpu_buf.data(), cache->up_expert_size);
            if (n == cache->up_expert_size) {
                size_t slot_off = (size_t)(layer_off_slot + slot) * cache->up_stride;
                raw_buf_write(cache->buf_up.get(), slot_off, cpu_buf.data(), n);
            }
        }

        // load down weights
        {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_down, layer, eid,
                                         cpu_buf.data(), cache->down_expert_size);
            if (n == cache->down_expert_size) {
                size_t slot_off = (size_t)(layer_off_slot + slot) * cache->down_stride;
                raw_buf_write(cache->buf_down.get(), slot_off, cpu_buf.data(), n);
            }
        }

        // update tracking
        cache->h_slot_of[layer_off_expert + eid] = slot;
        cache->h_expert_in[layer_off_slot + slot] = eid;
        cache->h_slot_used[layer_off_slot + slot] = 1;
        slot_map_changed = true;
    }

    // sync slot_map to GPU
    if (slot_map_changed && cache->buf_slot_map) {
        size_t layer_byte_off = (size_t)layer * n_experts * sizeof(int32_t);
        size_t layer_byte_sz  = (size_t)n_experts * sizeof(int32_t);
        raw_buf_write(cache->buf_slot_map.get(), layer_byte_off,
                      cache->h_slot_of.data() + layer_off_expert, layer_byte_sz);
    }
}

void dyn_ex_cache_fill(dyn_ex_cache * cache) {
    if (!cache) return;
    std::vector<int> ids(cache->n_slots);
    for (int l = 0; l < cache->n_layers; l++) {
        for (int s = 0; s < cache->n_slots; s++) ids[s] = s;
        dyn_ex_cache_ensure(cache, l, ids.data(), cache->n_slots);
    }
    LLAMA_LOG_INFO("dyn-ex: filled %d layers with %d experts each\n",
                   cache->n_layers, cache->n_slots);
}

void dyn_ex_cache_prefetch(dyn_ex_cache * cache, int layer, const int * expert_ids, int n_ids,
                           const float * scores) {
    if (!cache || !cache->copy_backend) return;
    if (layer < 0 || layer >= cache->n_layers) return;
    if (!expert_ids || n_ids <= 0) return;

    const int n_experts = cache->n_experts;
    const int n_slots   = cache->n_slots;
    const int layer_off_expert = layer * n_experts;
    const int layer_off_slot   = layer * n_slots;
    (void)scores; // eviction priority — used when predictor provides scores

    // ring-buffer staging index
    std::vector<uint8_t> & cpu_buf = cache->staging_bufs[cache->staging_idx];
    cache->staging_idx = (cache->staging_idx + 1) % (int)cache->staging_bufs.size();

    int next_slot = 0;
    for (int i = 0; i < n_ids; i++) {
        int eid = expert_ids[i];
        if (eid < 0 || eid >= n_experts) continue;

        // skip if already loaded
        if (cache->h_slot_of[layer_off_expert + eid] != DYN_EX_SENTINEL) continue;

        // find free slot
        int slot = -1;
        for (int attempt = 0; attempt < n_slots; attempt++) {
            int s = (next_slot + attempt) % n_slots;
            if (!cache->h_slot_used[layer_off_slot + s]) {
                slot = s;
                break;
            }
        }
        if (slot < 0) {
            // all slots occupied, skip prefetch for this expert
            continue;
        }
        next_slot = (slot + 1) % n_slots;

        int ev_idx = layer_off_slot + slot;
        // wait for previous copy on this slot
        if (cache->copy_events[ev_idx]) {
            ggml_backend_event_synchronize(cache->copy_events[ev_idx]);
        }

        // mark slot as in-use before async copy starts
        cache->h_slot_used[ev_idx] = 1;
        cache->h_expert_in[ev_idx] = eid;
        cache->h_slot_of[layer_off_expert + eid] = slot;

        // read gate_up weights from .bin into staging buffer
        if (cache->buf_gate_up && cache->pi_gate_up >= 0 && cache->t_gate_up[layer]) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_gate_up, layer, eid,
                                         cpu_buf.data(), cache->gate_up_expert_size);
            if (n == cache->gate_up_expert_size) {
                size_t slot_off = (size_t)slot * cache->gate_up_stride;
                ggml_backend_tensor_set_async(
                    cache->copy_backend,
                    cache->t_gate_up[layer], cpu_buf.data(), slot_off, n);
            }
        }
        if (cache->buf_gate && cache->pi_gate >= 0 && cache->t_gate[layer]) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_gate, layer, eid,
                                         cpu_buf.data(), cache->gate_expert_size);
            if (n == cache->gate_expert_size) {
                size_t slot_off = (size_t)slot * cache->gate_stride;
                ggml_backend_tensor_set_async(
                    cache->copy_backend,
                    cache->t_gate[layer], cpu_buf.data(), slot_off, n);
            }
        }
        if (cache->buf_up && cache->pi_up >= 0 && cache->t_up[layer]) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_up, layer, eid,
                                         cpu_buf.data(), cache->up_expert_size);
            if (n == cache->up_expert_size) {
                size_t slot_off = (size_t)slot * cache->up_stride;
                ggml_backend_tensor_set_async(
                    cache->copy_backend,
                    cache->t_up[layer], cpu_buf.data(), slot_off, n);
            }
        }

        // read down weights from .bin into staging buffer (reuse same buf serially)
        if (cache->buf_down && cache->pi_down >= 0 && cache->t_down[layer]) {
            size_t n = dyn_ex_read_param(cache->reader, cache->pi_down, layer, eid,
                                         cpu_buf.data(), cache->down_expert_size);
            if (n == cache->down_expert_size) {
                size_t slot_off = (size_t)slot * cache->down_stride;
                ggml_backend_tensor_set_async(
                    cache->copy_backend,
                    cache->t_down[layer], cpu_buf.data(), slot_off, n);
            }
        }

        // record event for this slot
        ggml_backend_event_record(cache->copy_events[ev_idx], cache->copy_backend);
    }
}

void dyn_ex_cache_wait(dyn_ex_cache * cache, int layer) {
    if (!cache || !cache->copy_backend) return;
    if (layer < 0 || layer >= cache->n_layers) return;

    int layer_off_slot = layer * cache->n_slots;
    for (int s = 0; s < cache->n_slots; s++) {
        int ev_idx = layer_off_slot + s;
        auto & ev = cache->copy_events[ev_idx];
        if (ev) {
            // check if event is done before synchronizing
            ggml_backend_event_synchronize(ev);
        }
    }
}

// ── predictor ──────────────────────────────────────────────────────────

#define DYN_EX_PRED_MAGIC "DXP2"

dyn_ex_predictor * dyn_ex_predictor_load(const char * path, int D, int L, int E, int H) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        LLAMA_LOG_ERROR("dyn-ex predictor: failed to open %s\n", path);
        return nullptr;
    }

    char magic[5] = {};
    int32_t hdr[4] = {};
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, DYN_EX_PRED_MAGIC, 4) != 0) {
        LLAMA_LOG_ERROR("dyn-ex predictor: bad magic\n");
        fclose(f);
        return nullptr;
    }
    if (fread(hdr, sizeof(hdr), 1, f) != 1) {
        LLAMA_LOG_ERROR("dyn-ex predictor: short header\n");
        fclose(f);
        return nullptr;
    }
    int fD = hdr[0], fL = hdr[1], fE = hdr[2], fH = hdr[3];
    if (fD != D || fL != L || fE != E || fH != H) {
        LLAMA_LOG_ERROR("dyn-ex predictor: dim mismatch: file (%d,%d,%d,%d) vs expected (%d,%d,%d,%d)\n",
                        fD, fL, fE, fH, D, L, E, H);
        fclose(f);
        return nullptr;
    }

    auto * p = new dyn_ex_predictor();
    p->D = D; p->L = L; p->E = E; p->H = H;

    auto load_vec = [&](std::vector<float> & v, size_t n) -> bool {
        v.resize(n);
        return fread(v.data(), sizeof(float), n, f) == n;
    };

    bool ok = true;
    ok = ok && load_vec(p->trunk_w, (size_t)H * D);
    ok = ok && load_vec(p->trunk_b, (size_t)H);
    ok = ok && load_vec(p->W1,      (size_t)L * H * (H + E));
    ok = ok && load_vec(p->b1,      (size_t)L * H);
    ok = ok && load_vec(p->W2,      (size_t)L * E * H);
    ok = ok && load_vec(p->b2,      (size_t)L * E);

    fclose(f);

    if (!ok) {
        LLAMA_LOG_ERROR("dyn-ex predictor: short weight data\n");
        delete p;
        return nullptr;
    }

    LLAMA_LOG_INFO("dyn-ex predictor: loaded %s (D=%d L=%d E=%d H=%d)\n", path, D, L, E, H);
    return p;
}

void dyn_ex_predictor_free(dyn_ex_predictor * p) {
    delete p;
}

void dyn_ex_predictor_predict(dyn_ex_predictor * p,
    int n_tokens,
    const float * ht, const float * Et, int layer, int top_k,
    int * top_ids, float * scores) {

    if (!p || n_tokens <= 0 || !ht || !Et || !top_ids) return;

    const int D = p->D, E = p->E, H = p->H;
    const int HpE = H + E;

    std::vector<float> z(n_tokens * H);
    std::vector<float> x(n_tokens * HpE);
    std::vector<float> h_buf(n_tokens * H);
    std::vector<float> logits(n_tokens * E);

    for (int n = 0; n < n_tokens; n++) {
        for (int j = 0; j < H; j++) {
            float sum = p->trunk_b[j];
            for (int i = 0; i < D; i++) {
                sum += ht[n * D + i] * p->trunk_w[j * D + i];
            }
            z[n * H + j] = sum > 0.0f ? sum : 0.0f;
        }
    }

    for (int n = 0; n < n_tokens; n++) {
        memcpy(x.data() + n * HpE, z.data() + n * H, H * sizeof(float));
        memcpy(x.data() + n * HpE + H, Et + n * E, E * sizeof(float));
    }

    const float * W1_l = p->W1.data() + (size_t)layer * H * HpE;
    const float * b1_l = p->b1.data() + (size_t)layer * H;

    for (int n = 0; n < n_tokens; n++) {
        for (int j = 0; j < H; j++) {
            float sum = b1_l[j];
            for (int i = 0; i < HpE; i++) {
                sum += x[n * HpE + i] * W1_l[j * HpE + i];
            }
            h_buf[n * H + j] = sum > 0.0f ? sum : 0.0f;
        }
    }

    const float * W2_l = p->W2.data() + (size_t)layer * E * H;
    const float * b2_l = p->b2.data() + (size_t)layer * E;

    for (int n = 0; n < n_tokens; n++) {
        for (int j = 0; j < E; j++) {
            float sum = b2_l[j];
            for (int i = 0; i < H; i++) {
                sum += h_buf[n * H + i] * W2_l[j * H + i];
            }
            logits[n * E + j] = sum;
        }
    }

    if (scores) {
        memcpy(scores, logits.data(), n_tokens * E * sizeof(float));
    }

    for (int n = 0; n < n_tokens; n++) {
        const float * row = logits.data() + n * E;
        int * out = top_ids + n * top_k;

        std::vector<std::pair<float, int>> pairs(E);
        for (int e = 0; e < E; e++) {
            pairs[e] = {row[e], e};
        }
        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        for (int k = 0; k < top_k; k++) {
            out[k] = pairs[k].second;
        }
    }
}

void dyn_ex_cache_alloc_barriers(dyn_ex_cache * cache, ggml_backend_dev_t dev, int n_layers, int n_expert_used) {
    int n_elements = n_expert_used * 4096;
    int64_t total = n_elements + 2;
    auto * dev_buft = ggml_backend_dev_buffer_type(dev);  // DEVICE buffer type

    for (int i = 0; i < n_layers; i++) {
        auto buf = ggml_backend_buft_alloc_buffer(dev_buft, total * sizeof(int32_t));
        if (!buf) continue;
        auto * t = new ggml_tensor();
        memset(t, 0, sizeof(ggml_tensor));
        t->type = GGML_TYPE_I32;
        t->ne[0] = total; t->ne[1] = 1; t->ne[2] = 1; t->ne[3] = 1;
        t->nb[0] = sizeof(int32_t); t->nb[1] = sizeof(int32_t) * total; t->nb[2] = t->nb[1]; t->nb[3] = t->nb[2];
        ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));
        cache->buf_barrier.push_back(ggml_backend_buffer_ptr(buf));
        cache->t_barrier.push_back(t);
    }
}
