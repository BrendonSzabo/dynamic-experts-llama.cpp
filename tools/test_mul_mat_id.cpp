// Standalone MUL_MAT_ID kernel test
// Build: cmake --build build --target <this>
// Run:   ./build/bin/test_mul_mat_id
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>

// Generate known weight matrix: expert e, position (r,c) = e*100 + r*10 + c
static void fill_weights(float * data, int ne0, int ne1, int ne2) {
    for (int e = 0; e < ne2; e++)
        for (int r = 0; r < ne0; r++)
            for (int c = 0; c < ne1; c++)
                data[(e * ne1 + c) * ne0 + r] = (float)(e * 100 + r * 10 + c);
}

// Generate known input: token t, position i = (t+1)*10 + i
static void fill_input(float * data, int ne0, int ne2) {
    for (int t = 0; t < ne2; t++)
        for (int i = 0; i < ne0; i++)
            data[t * ne0 + i] = (float)((t+1) * 10 + i);
}

static float compute_dot(const float * w, const float * x, int n, int w_stride) {
    float sum = 0;
    for (int j = 0; j < n; j++) sum += w[j * w_stride] * x[j];
    return sum;
}

int main() {
    ggml_backend_load_all();

    // Dimensions: 2 experts, 2 tokens, ne0=4, ne1=3
    const int ne0 = 4; // output dimension
    const int ne1 = 3; // input dimension
    const int ne2 = 2; // 2 experts
    const int n_expert_used = 2;
    const int n_tokens = 2;

    // Init ggml context with no_alloc (we use backend buffers)
    ggml_init_params ctx_params = { 32*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ctx_params);

    // Create weight tensor [ne0, ne1, ne2]
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(w, "weights");

    // Create input tensor [ne0, 1, n_tokens]
    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, 1, n_tokens);
    ggml_set_name(inp, "input");

    // Create ids tensor [n_expert_used, n_tokens]
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_set_name(ids, "ids");

    // Create dst tensor [ne1, n_expert_used, n_tokens]
    ggml_tensor * dst = ggml_mul_mat_id(ctx, w, inp, ids);
    ggml_set_name(dst, "dst");

    // Build graph and allocate
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);

    // Fill weight data
    std::vector<float> w_data(ne0 * ne1 * ne2);
    fill_weights(w_data.data(), ne0, ne1, ne2);
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size() * sizeof(float));

    // Fill input data
    std::vector<float> inp_data(ne0 * n_tokens);
    fill_input(inp_data.data(), ne0, n_tokens);
    ggml_backend_tensor_set(inp, inp_data.data(), 0, inp_data.size() * sizeof(float));

    // Fill ids data
    std::vector<int32_t> ids_data = {0, 1, 1, 0};
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));

    // Print weights
    printf("=== WEIGHTS (ne0=%d ne1=%d ne2=%d) ===\n", ne0, ne1, ne2);
    for (int e = 0; e < ne2; e++) {
        printf("Expert %d:\n", e);
        const float * ww = w_data.data() + e * ne0 * ne1;
        for (int r = 0; r < ne0; r++) {
            printf("  row %d: [", r);
            for (int c = 0; c < ne1; c++) printf(" %6.0f", ww[c * ne0 + r]);
            printf(" ]\n");
        }
    }

    // Print input
    printf("\n=== INPUT (ne0=%d n_tokens=%d) ===\n", ne0, n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        printf("Token %d: [", t);
        for (int i = 0; i < ne0; i++) printf(" %6.0f", inp_data[t * ne0 + i]);
        printf(" ]\n");
    }

    // Print ids
    printf("\n=== IDS ===\n");
    for (int t = 0; t < n_tokens; t++) {
        printf("Token %d: [", t);
        for (int e = 0; e < n_expert_used; e++) printf(" %d", ids_data[e * n_tokens + t]);
        printf(" ]\n");
    }

    // Compute expected output by hand
    printf("\n=== EXPECTED (computed by hand) ===\n");
    printf("dst shape: [ne0=%d, n_expert_used=%d, n_tokens=%d]\n", ne1, n_expert_used, n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            int eid = ids_data[e * n_tokens + t];
            const float * ww = w_data.data() + eid * ne0 * ne1;
            printf("token=%d expert=%d (eid=%d): [", t, e, eid);
            for (int r = 0; r < ne1; r++) {
                float val = compute_dot(ww + r, &inp_data[t * ne0], ne0, ne1);
                printf(" %8.1f", val);
            }
            printf(" ]\n");
        }
    }

    // Run kernel on CPU
    printf("\n=== KERNEL OUTPUT ===\n");
    ggml_backend_graph_compute(cpu_backend, gf);

    std::vector<float> dst_data(ne1 * n_expert_used * n_tokens);
    ggml_backend_tensor_get(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));

    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            int eid = ids_data[e * n_tokens + t];
            printf("token=%d expert=%d (eid=%d): [", t, e, eid);
            for (int r = 0; r < ne1; r++) {
                float val = dst_data[t * n_expert_used * ne1 + e * ne1 + r];
                printf(" %8.1f", val);
            }
            printf(" ]\n");
        }
    }

    // Compare
    printf("\n=== COMPARISON ===\n");
    printf("Compare EXPECTED vs KERNEL OUTPUT above - they should match\n");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);
    return 0;
}
