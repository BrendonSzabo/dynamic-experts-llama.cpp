#include "llama-dyn-ex.h"
#include "llama-impl.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime_api.h>
#endif

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
        case 1:  return GGML_TYPE_F16;
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

    // pre-fault pages into OS page cache so .bin reads don't block on disk I/O
    madvise(addr, file_size, MADV_WILLNEED);
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
        size_t per_expert_bytes = (size_t)numel * ggml_type_size(r->params[i].type) / ggml_blck_size(r->params[i].type);
        size_t stride           = round_up_page(per_expert_bytes);

        r->param_data_off[i] = data_off;
        r->param_stride[i]   = stride;
        r->param_size[i]     = per_expert_bytes;

        data_off += (size_t)r->n_layers * (size_t)r->n_experts * stride;
    }

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

    size_t expert_size = r->param_size[param_idx];

    size_t file_off = r->param_data_off[param_idx]
                    + (size_t)(layer * r->n_experts + expert_id) * r->param_stride[param_idx];

    size_t n = expert_size < buf_size ? expert_size : buf_size;
    memcpy(buf, (const uint8_t *)r->mmap_addr + file_off, n);
    return n;
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
#ifdef GGML_USE_CUDA
    static cudaStream_t s = nullptr;
    if (!s) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    void * dst = (char *)ggml_backend_buffer_get_base(buf) + offset;
    cudaMemcpyAsync(dst, data, size, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
#else
    ggml_tensor dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.buffer = buf;
    dummy.data   = (char *)ggml_backend_buffer_get_base(buf) + offset;
    dummy.type   = GGML_TYPE_I8;
    dummy.ne[0]  = (int64_t)size;
    dummy.ne[1]  = 1; dummy.ne[2] = 1; dummy.ne[3] = 1;
    dummy.nb[0]  = 1;
    ggml_backend_tensor_set(&dummy, data, 0, size);
#endif
}

dyn_ex_cache * dyn_ex_cache_init(
    dyn_ex_reader * reader,
    int n_l1, int n_l2, int n_expert_used,
    ggml_backend_dev_t dev) {

    if (!reader || n_l1 < 1) return nullptr;
    if ((n_l1 & (n_l1 - 1)) != 0) {
        LLAMA_LOG_ERROR("dyn-ex: n_l1 must be power of 2, got %d\n", n_l1);
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

    auto * cache = new dyn_ex_cache();
    cache->reader        = reader;
    cache->n_l1          = n_l1;
    cache->n_l2          = n_l2;
    cache->n_layers      = n_layers;
    cache->n_experts     = n_experts;
    cache->n_expert_used = n_expert_used;

    cache->n_ubatch = std::max(1, (n_l1 / n_expert_used) / 2);
    cache->n_hot    = cache->n_ubatch * n_expert_used;
    cache->n_cache  = n_l1 - cache->n_hot;

    cache->pi_gate_up = pi_gate_up;
    cache->pi_gate    = pi_gate;
    cache->pi_up      = pi_up;
    cache->pi_down    = pi_down;

    cache->l1_layer.assign(n_l1, DYN_EX_SENTINEL);
    cache->l1_expert.assign(n_l1, DYN_EX_SENTINEL);
    cache->l1_age.assign(n_l1, 0);
    cache->l1_in_use.assign(n_l1, 0);

    cache->l2.resize(n_layers);

    LLAMA_LOG_INFO("dyn-ex: cache init: %d layers, %d experts, L1=%d slots (hot=%d cache=%d ubatch=%d), L2=%d per layer\n",
                   n_layers, n_experts, n_l1, cache->n_hot, cache->n_cache, cache->n_ubatch, n_l2);

    (void)dev;
    return cache;
}

void dyn_ex_cache_set_layer_size(
    dyn_ex_cache * cache, int layer,
    size_t gate_size, size_t gate_row, size_t up_size, size_t up_row,
    size_t down_size, size_t down_row, size_t gate_up_size, size_t gate_up_row) {

    if (!cache || layer < 0 || layer >= cache->n_layers) return;
    auto & l2 = cache->l2[layer];
    l2.gate_size    = gate_size;
    l2.gate_row     = gate_row;
    l2.up_size      = up_size;
    l2.up_row       = up_row;
    l2.down_size    = down_size;
    l2.down_row     = down_row;
    l2.gate_up_size = gate_up_size;
    l2.gate_up_row  = gate_up_row;

    if (cache->n_l2 > 0) {
        l2.expert.assign(cache->n_l2, DYN_EX_SENTINEL);
        l2.slot_of.assign(cache->n_experts, DYN_EX_SENTINEL);
        l2.age.assign(cache->n_l2, 0);
        if (gate_size)    l2.gate.resize((size_t)cache->n_l2 * gate_size);
        if (up_size)   l2.up.resize((size_t)cache->n_l2 * up_size);
        if (down_size) l2.down.resize((size_t)cache->n_l2 * down_size);
        if (gate_up_size) l2.gate_up.resize((size_t)cache->n_l2 * gate_up_size);
    }
}

void dyn_ex_cache_free(dyn_ex_cache * cache) {
    if (!cache) return;
#ifdef GGML_USE_CUDA
    for (auto * hp : cache->t_barrier_host) {
        if (hp) cudaFreeHost(hp);
    }
#endif
    delete cache->reader;
    delete cache;
}

void dyn_ex_cache_alloc_barriers(dyn_ex_cache * cache, int n_layers, int n_expert_used) {
    int max_el = n_expert_used * 4096;
    for (int i = 0; i < n_layers; i++) {
        auto * t = new ggml_tensor();
        memset(t, 0, sizeof(ggml_tensor));
        t->type = GGML_TYPE_I32;
        t->ne[0] = max_el + 2; t->ne[1] = 1; t->ne[2] = 1; t->ne[3] = 1;
        t->nb[0] = sizeof(int32_t); t->nb[1] = (size_t)sizeof(int32_t) * (max_el + 2);
        t->nb[2] = t->nb[1]; t->nb[3] = t->nb[2];
        t->op_params[0] = i;
        t->flags = GGML_TENSOR_FLAG_EXTERNAL | GGML_TENSOR_FLAG_COMPUTE;
        cache->t_barrier.push_back(t);
        cache->t_barrier_host.push_back(nullptr);
    }
}
