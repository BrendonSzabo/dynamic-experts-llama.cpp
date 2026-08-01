#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "llama-dyn-ex.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <weights.bin>\n", argv[0]); return 1; }
    const char *path = argv[1];

    dyn_ex_reader *reader = dyn_ex_reader_open(path);
    if (!reader) { fprintf(stderr, "dyn_ex_reader_open failed\n"); return 1; }

    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    void *addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    const uint8_t *raw = (const uint8_t *)addr;

    int errors = 0, checked = 0;
    size_t max_esz = 0;
    for (int pi = 0; pi < reader->n_params; pi++) {
        size_t s = dyn_ex_param_size(reader, pi);
        if (s > max_esz) max_esz = s;
    }

    std::vector<uint8_t> rbuf(max_esz), mbuf(max_esz);

    for (int pi = 0; pi < reader->n_params; pi++) {
        size_t esz = dyn_ex_param_size(reader, pi);
        const char *pname = reader->params[pi].name;

        for (int il = 0; il < reader->n_layers; il++) {
            for (int eid = 0; eid < reader->n_experts; eid++) {
                checked++;
                size_t n = dyn_ex_read_param(reader, pi, il, eid, rbuf.data(), esz);
                size_t off = reader->param_data_off[pi] + (size_t)(il*reader->n_experts+eid)*reader->param_stride[pi];
                memcpy(mbuf.data(), raw + off, esz);
                if (n != esz || memcmp(rbuf.data(), mbuf.data(), esz) != 0) {
                    errors++;
                    fprintf(stderr, "MISMATCH L%d e%d %s: n=%zu\n", il, eid, pname, n);
                    if (errors >= 10) goto done;
                }
            }
        }
        fprintf(stderr, "  %s: %d/%d ok\n", pname, reader->n_layers * reader->n_experts - errors, reader->n_layers * reader->n_experts);
    }
done:
    fprintf(stderr, "\nChecked %d, errors %d\n", checked, errors);
    dyn_ex_reader_close(reader);
    munmap(addr, st.st_size);
    close(fd);
    return errors ? 1 : 0;
}
