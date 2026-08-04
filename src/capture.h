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
    size_t        buf_offset;
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

    bool   is_active() const { return m_active; }
    size_t bytes_per_token() const { return m_bytes_per_token; }

    void begin_token(int token_id, int position);
    void launch_dma();
    void sync_and_flush();
    void reset_entries();

private:
    FILE * m_fp = nullptr;
    bool   m_active = false;

    uint8_t * m_pinned[2] = {nullptr, nullptr};
    int m_active_buf = 0;

    size_t m_bytes_per_token = 0;
    std::vector<capture_entry> m_entries;

    int m_token_id = 0;
    int m_position = 0;
    int m_tokens_written = 0;
    bool m_header_written = false;

    void write_header(int n_embd, int n_vocab, int n_layer, int n_experts, int n_expert_used);
    void flush_current();
};
