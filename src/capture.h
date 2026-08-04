#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct capture_entry {
    ggml_tensor * tensor;
    std::string   name;
    int           il;
    int64_t       ne[4];
    int           n_dims;
};

class CaptureCollector {
public:
    bool init(const char * dir, std::string filename,
              int n_embd, int n_vocab, int n_layer,
              int n_experts, int n_expert_used);
    void close();

    void register_tensor(ggml_tensor * t, const char * name, int il);

    bool is_active() const { return m_active; }

    void begin_token(int token_id, int position);
    void launch_dma();
    void sync_and_flush();
    void reset_entries();

private:
    static constexpr size_t CHUNK_SIZE = 64 * 1024 * 1024;

    struct pinned_chunk {
        uint8_t * data = nullptr;
        size_t    used = 0;
        pinned_chunk * next = nullptr;
    };

    FILE * m_fp = nullptr;
    bool   m_active = false;

    void * m_stream = nullptr;

    pinned_chunk * m_chunk_head = nullptr;
    pinned_chunk * m_chunk_tail = nullptr;
    size_t m_chunk_offset = 0;

    std::vector<capture_entry> m_entries;

    int m_token_id = 0;
    int m_position = 0;
    int m_tokens_written = 0;
    bool m_header_written = false;

    void write_header(int n_embd, int n_vocab, int n_layer, int n_experts, int n_expert_used);
    void flush_current();
    void free_chunks();
    uint8_t * alloc_dma_space(size_t nb);
};
