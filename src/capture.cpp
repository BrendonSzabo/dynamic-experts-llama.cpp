#include "capture.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstring>
#include <ctime>

static constexpr uint32_t CAPTURE_MAGIC   = 0x44434552; // "RECD"
static constexpr uint32_t CAPTURE_VERSION = 1;
static constexpr size_t   PINNED_BUF_SIZE = 256 * 1024 * 1024; // 256 MB per buffer

static void write_u32(FILE * fp, uint32_t v) { fwrite(&v, sizeof(v), 1, fp); }
static void write_i32(FILE * fp, int32_t v)  { fwrite(&v, sizeof(v), 1, fp); }
static void write_i64(FILE * fp, int64_t v)  { fwrite(&v, sizeof(v), 1, fp); }

void CaptureCollector::write_header(int n_embd, int n_vocab, int n_layer, int n_experts, int n_expert_used) {
    write_u32(m_fp, CAPTURE_MAGIC);
    write_u32(m_fp, CAPTURE_VERSION);
    write_u32(m_fp, (uint32_t)n_embd);
    write_u32(m_fp, (uint32_t)n_vocab);
    write_u32(m_fp, (uint32_t)n_layer);
    write_u32(m_fp, (uint32_t)n_experts);
    write_u32(m_fp, (uint32_t)n_expert_used);
    fflush(m_fp);
    m_header_written = true;
}

bool CaptureCollector::init(const char * dir, std::string filename,
                             int n_embd, int n_vocab, int n_layer,
                             int n_experts, int n_expert_used) {
#ifdef GGML_USE_CUDA
    std::string path = std::string(dir) + "/" + filename;
    m_fp = fopen(path.c_str(), "wb");
    if (!m_fp) return false;

    write_header(n_embd, n_vocab, n_layer, n_experts, n_expert_used);

    for (int i = 0; i < 2; i++) {
        cudaError_t err = cudaMallocHost(&m_pinned[i], PINNED_BUF_SIZE);
        if (err != cudaSuccess) {
            if (i > 0) cudaFreeHost(m_pinned[0]);
            fclose(m_fp);
            m_fp = nullptr;
            return false;
        }
    }

    m_active = true;
    return true;
#else
    (void)dir; (void)filename; (void)n_embd; (void)n_vocab;
    (void)n_layer; (void)n_experts; (void)n_expert_used;
    return false;
#endif
}

void CaptureCollector::close() {
    if (!m_active) return;
#ifdef GGML_USE_CUDA
    for (int i = 0; i < 2; i++) {
        if (m_pinned[i]) { cudaFreeHost(m_pinned[i]); m_pinned[i] = nullptr; }
    }
#endif
    if (m_fp) { fclose(m_fp); m_fp = nullptr; }
    m_active = false;
}

void CaptureCollector::register_tensor(ggml_tensor * t, const char * name, int il) {
    if (!m_active || !t) return;
    if (t->op == GGML_OP_NONE) return;
    if (t->view_src != nullptr) return;
    if (t->op == GGML_OP_DYN_EX_BARRIER) return;

    capture_entry e;
    e.tensor     = t;
    e.name       = name;
    e.il         = il;
    e.n_dims     = ggml_n_dims(t);
    for (int d = 0; d < GGML_MAX_DIMS; d++) e.ne[d] = t->ne[d];

    e.buf_offset = m_bytes_per_token;
    m_bytes_per_token += ggml_nbytes(t);

    m_entries.push_back(e);
}

void CaptureCollector::reset_entries() {
    m_entries.clear();
    m_bytes_per_token = 0;
}

void CaptureCollector::begin_token(int token_id, int position) {
    m_token_id = token_id;
    m_position = position;
    m_active_buf = 1 - m_active_buf;
}

void CaptureCollector::launch_dma() {
    if (!m_active) return;
#ifdef GGML_USE_CUDA
    uint8_t * dst = m_pinned[m_active_buf];
    size_t offset = 0;
    for (auto & e : m_entries) {
        if (!e.tensor->data) continue;
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, e.tensor->data) != cudaSuccess) continue;
        if (attr.type != cudaMemoryTypeDevice) continue;
        size_t nb = ggml_nbytes(e.tensor);
        if (offset + nb > PINNED_BUF_SIZE) {
            fprintf(stderr, "capture: pinned buffer overflow (%zu + %zu > %zu), disabling\n",
                    offset, nb, PINNED_BUF_SIZE);
            m_active = false;
            return;
        }
        cudaMemcpy(dst + offset, e.tensor->data, nb, cudaMemcpyDeviceToHost);
        offset += nb;
    }
#endif
}

void CaptureCollector::sync_and_flush() {
    if (!m_active) return;
    flush_current();
}

void CaptureCollector::flush_current() {
    if (!m_active || !m_fp) return;

    uint32_t n_valid = 0;
    for (auto & e : m_entries) {
        if (!e.tensor->data) continue;
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, e.tensor->data) != cudaSuccess) continue;
        if (attr.type != cudaMemoryTypeDevice) continue;
        n_valid++;
    }

    write_i32(m_fp, m_token_id);
    write_i32(m_fp, m_position);
    write_u32(m_fp, n_valid);

    uint8_t * src = m_pinned[m_active_buf];
    size_t src_offset = 0;

    for (auto & e : m_entries) {
        if (!e.tensor->data) continue;
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, e.tensor->data) != cudaSuccess) continue;
        if (attr.type != cudaMemoryTypeDevice) continue;
        size_t nb = ggml_nbytes(e.tensor);
        uint32_t name_len = (uint32_t)e.name.size();
        write_u32(m_fp, name_len);
        fwrite(e.name.data(), 1, name_len, m_fp);
        write_i32(m_fp, e.il);
        write_u32(m_fp, (uint32_t)e.n_dims);
        for (int d = 0; d < e.n_dims; d++) write_i64(m_fp, e.tensor->ne[d]);
        write_u32(m_fp, (uint32_t)e.tensor->type);

        int64_t nelements = 1;
        for (int d = 0; d < e.n_dims; d++) nelements *= e.tensor->ne[d];

        fwrite(src + src_offset, ggml_type_size(e.tensor->type), nelements, m_fp);
        src_offset += nb;
    }

    fflush(m_fp);
    m_tokens_written++;
}
