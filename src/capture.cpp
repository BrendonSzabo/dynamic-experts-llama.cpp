#include "capture.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstring>

static constexpr uint32_t CAPTURE_MAGIC   = 0x44434552;
static constexpr uint32_t CAPTURE_VERSION = 1;

static void write_u32(FILE * fp, uint32_t v) { fwrite(&v, sizeof(v), 1, fp); }
static void write_i32(FILE * fp, int32_t v)  { fwrite(&v, sizeof(v), 1, fp); }
static void write_i64(FILE * fp, int64_t v)  { fwrite(&v, sizeof(v), 1, fp); }

uint8_t * CaptureCollector::alloc_dma_space(size_t nb) {
    if (!m_chunk_tail || m_chunk_offset + nb > CHUNK_SIZE) {
        auto * c = new pinned_chunk();
#ifdef GGML_USE_CUDA
        cudaMallocHost(&c->data, CHUNK_SIZE);
#endif
        c->used = 0;
        c->next = nullptr;
        if (m_chunk_tail) m_chunk_tail->next = c;
        else              m_chunk_head = c;
        m_chunk_tail = c;
        m_chunk_offset = 0;
    }
    uint8_t * ptr = m_chunk_tail->data + m_chunk_offset;
    m_chunk_offset += nb;
    m_chunk_tail->used = m_chunk_offset;
    return ptr;
}

void CaptureCollector::free_chunks() {
    while (m_chunk_head) {
        auto * next = m_chunk_head->next;
#ifdef GGML_USE_CUDA
        if (m_chunk_head->data) cudaFreeHost(m_chunk_head->data);
#endif
        delete m_chunk_head;
        m_chunk_head = next;
    }
    m_chunk_tail = nullptr;
}

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
    cudaStreamCreate((cudaStream_t*)&m_stream);
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
    if (m_stream) {
        cudaStreamSynchronize((cudaStream_t)m_stream);
        cudaStreamDestroy((cudaStream_t)m_stream);
        m_stream = nullptr;
    }
#endif
    free_chunks();
    if (m_fp) { fclose(m_fp); m_fp = nullptr; }
    m_active = false;
}

void CaptureCollector::register_tensor(ggml_tensor * t, const char * name, int il) {
    if (!m_active || !t) return;
    if (t->op == GGML_OP_NONE) return;
    if (t->view_src != nullptr) return;
    if (t->op == GGML_OP_DYN_EX_BARRIER) return;

    capture_entry e;
    e.tensor = t;
    e.name   = name;
    e.il     = il;
    e.n_dims = ggml_n_dims(t);
    for (int d = 0; d < GGML_MAX_DIMS; d++) e.ne[d] = t->ne[d];
    m_entries.push_back(e);
}

void CaptureCollector::reset_entries() {
    m_entries.clear();
}

void CaptureCollector::begin_token(int token_id, int position) {
    m_token_id = token_id;
    m_position = position;
}

void CaptureCollector::launch_dma() {
    if (!m_active) return;
#ifdef GGML_USE_CUDA
    for (auto & e : m_entries) {
        if (!e.tensor->data) continue;
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, e.tensor->data) != cudaSuccess) continue;
        if (attr.type != cudaMemoryTypeDevice) continue;
        size_t nb = ggml_nbytes(e.tensor);
        uint8_t * dst = alloc_dma_space(nb);
        cudaMemcpyAsync(dst, e.tensor->data, nb, cudaMemcpyDeviceToHost, (cudaStream_t)m_stream);
    }
#endif
}

void CaptureCollector::sync_and_flush() {
    if (!m_active) return;
#ifdef GGML_USE_CUDA
    cudaStreamSynchronize((cudaStream_t)m_stream);
#endif
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

    pinned_chunk * chunk = m_chunk_head;
    size_t chunk_pos = 0;

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

        size_t remaining = nb;
        while (remaining > 0 && chunk) {
            size_t avail = chunk->used - chunk_pos;
            size_t n = std::min(remaining, avail);
            fwrite(chunk->data + chunk_pos, 1, n, m_fp);
            chunk_pos += n;
            remaining -= n;
            if (chunk_pos >= chunk->used) {
                chunk = chunk->next;
                chunk_pos = 0;
            }
        }
    }

    fflush(m_fp);
    m_tokens_written++;

    free_chunks();
    m_chunk_offset = 0;
}
