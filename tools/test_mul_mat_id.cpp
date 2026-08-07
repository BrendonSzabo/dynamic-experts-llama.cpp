// Standalone MUL_MAT_ID kernel test — CPU vs CUDA comparison
// Build: cmake --build build --target test_mul_mat_id
// Run:   ./build/bin/test_mul_mat_id
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float compute_dot(const float * w, const float * x, int n) {
    float sum = 0;
    for (int j = 0; j < n; j++) sum += w[j] * x[j];
    return sum;
}

struct test_case {
    const char * desc;
    int ne0, ne1, ne2;       // weight: [ne0, ne1, ne2]  (K, N, n_experts or n_slots)
    int n_expert_used, n_tokens; // ids: [n_expert_used, n_tokens]
    ggml_type wtype;
    std::vector<int32_t> ids_data;
};

static bool run_test(const test_case & tc) {
    printf("\n=== %s ===\n", tc.desc);
    printf("weight: [%d, %d, %d] type=%s  ids: [%d, %d]\n",
           tc.ne0, tc.ne1, tc.ne2, ggml_type_name(tc.wtype),
           tc.n_expert_used, tc.n_tokens);

    printf("ids: [");
    for (size_t i = 0; i < tc.ids_data.size(); i++)
        printf("%d%s", tc.ids_data[i], i+1 < tc.ids_data.size() ? " " : "");
    printf("]\n");

    // prepare host data
    size_t w_bytes = tc.ne0 * tc.ne1 * tc.ne2 * ggml_type_size(tc.wtype) / ggml_blck_size(tc.wtype);
    if (tc.wtype == GGML_TYPE_F32) w_bytes = tc.ne0 * tc.ne1 * tc.ne2 * sizeof(float);
    else if (tc.wtype == GGML_TYPE_F16) w_bytes = tc.ne0 * tc.ne1 * tc.ne2 * sizeof(ggml_fp16_t);

    std::vector<float> wf32(tc.ne0 * tc.ne1 * tc.ne2);
    for (int e = 0; e < tc.ne2; e++)
        for (int c = 0; c < tc.ne1; c++)
            for (int r = 0; r < tc.ne0; r++)
                wf32[(e * tc.ne1 + c) * tc.ne0 + r] =
                    (float)(e * 100 + r * 10 + c);

    std::vector<uint8_t> w_host(w_bytes);
    if (tc.wtype == GGML_TYPE_F32) {
        memcpy(w_host.data(), wf32.data(), w_bytes);
    } else if (tc.wtype == GGML_TYPE_F16) {
        ggml_fp16_t * f16 = (ggml_fp16_t *)w_host.data();
        for (size_t i = 0; i < wf32.size(); i++)
            f16[i] = ggml_fp32_to_fp16(wf32[i]);
    } else {
        printf("SKIP: quant type not supported\n");
        return true;
    }

    std::vector<float> inp_data(tc.ne0 * tc.n_tokens);
    for (int t = 0; t < tc.n_tokens; t++)
        for (int i = 0; i < tc.ne0; i++)
            inp_data[t * tc.ne0 + i] = (float)((t+1) * 10 + i);

    // compute expected values by hand
    std::vector<float> expected(tc.ne1 * tc.n_expert_used * tc.n_tokens);
    for (int t = 0; t < tc.n_tokens; t++) {
        for (int e = 0; e < tc.n_expert_used; e++) {
            int slot = tc.ids_data[t * tc.n_expert_used + e];
            const float * ww = wf32.data() + slot * tc.ne0 * tc.ne1;
            for (int r = 0; r < tc.ne1; r++) {
                expected[t * tc.n_expert_used * tc.ne1 + e * tc.ne1 + r] =
                    compute_dot(ww + r * tc.ne0, &inp_data[t * tc.ne0], tc.ne0);
            }
        }
    }

    auto test_backend = [&](const char * name) -> bool {
        ggml_backend_t be = ggml_backend_init_by_name(name, nullptr);
        if (!be) { printf("  %s: SKIP (not available)\n", name); return true; }

        ggml_init_params ctx_params = { 32*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);

        ggml_tensor * w   = ggml_new_tensor_3d(ctx, tc.wtype, tc.ne0, tc.ne1, tc.ne2);
        ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.ne0, 1, tc.n_tokens);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_expert_used, tc.n_tokens);
        ggml_tensor * dst = ggml_mul_mat_id(ctx, w, inp, ids);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(w,   w_host.data(),   0, w_bytes);
        ggml_backend_tensor_set(inp, inp_data.data(), 0, inp_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, tc.ids_data.data(), 0, tc.ids_data.size() * sizeof(int32_t));

        ggml_backend_graph_compute(be, gf);

        std::vector<float> out(tc.ne1 * tc.n_expert_used * tc.n_tokens);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);

        // compare
        float max_diff = 0;
        int bad_count  = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = fabsf(expected[i] - out[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-3f) {
                if (bad_count < 5)
                    printf("  %s MISMATCH[%zu]: expected=%.6f got=%.6f diff=%.6f\n",
                           name, i, expected[i], out[i], diff);
                bad_count++;
            }
        }

        if (bad_count > 0) {
            printf("  %s: FAIL — %d mismatches, max_diff=%.6f\n", name, bad_count, max_diff);
            return false;
        }
        printf("  %s: PASS (max_diff=%.9f)\n", name, max_diff);
        return true;
    };

    bool ok = test_backend("CPU");
    ok = test_backend("CUDA0") && ok;
    return ok;
}

int main() {
    ggml_backend_load_all();

    bool all_pass = true;

    // Test 1: basic small test (original)
    all_pass &= run_test({"basic [2 experts, 2 tokens, f32]",
        4, 3, 2, 2, 2, GGML_TYPE_F32,
        {0, 1, 1, 0}});

    // Test 2: dyn-ex simulation — 16 slots, 2 tokens, 4 experts used per token
    all_pass &= run_test({"dyn-ex: 16 slots, slot-id indexing, f32",
        4, 3, 16, 4, 2, GGML_TYPE_F32,
        {0, 1, 2, 3, 4, 5, 6, 7}});

    // Test 3: dyn-ex with dedup — shared slots across tokens
    all_pass &= run_test({"dyn-ex: 16 slots, dedup slots across tokens, f32",
        4, 3, 16, 4, 2, GGML_TYPE_F32,
        {0, 1, 2, 3, 0, 1, 4, 5}});

    // Test 4: larger dimensions
    all_pass &= run_test({"dyn-ex: large dims, 32 slots, f32",
        64, 32, 32, 4, 1, GGML_TYPE_F32,
        {0, 1, 2, 3}});

    // Test 5: f16 weights with dyn-ex layout
    all_pass &= run_test({"dyn-ex: 16 slots, f16 weights",
        4, 3, 16, 4, 2, GGML_TYPE_F16,
        {0, 1, 2, 3, 4, 5, 6, 7}});

    // Test 6: touch boundary — ids pointing to last slot
    all_pass &= run_test({"dyn-ex: boundary — ids at last slot, f32",
        4, 3, 16, 2, 1, GGML_TYPE_F32,
        {0, 15}});

    // Test 7: ne2=256 vs ne2=8 — same weights (first 8), same ids (0-7)
    //   Verifies MUL_MAT_ID computes identically regardless of ne2 size
    {
        const int ne0 = 64, ne1 = 32, n_eu = 8, n_t = 1;
        const int ne2_large = 256, ne2_small = 8;
        const std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 6, 7};

        // weight data: first 8 experts identical, rest of ne2_large different
        std::vector<float> w_large(ne0 * ne1 * ne2_large);
        std::vector<float> w_small(ne0 * ne1 * ne2_small);
        for (int e = 0; e < ne2_large; e++)
            for (int c = 0; c < ne1; c++)
                for (int r = 0; r < ne0; r++)
                    w_large[(e * ne1 + c) * ne0 + r] = (float)(e * 100 + r * 10 + c);
        memcpy(w_small.data(), w_large.data(), ne0 * ne1 * 8 * sizeof(float));

        std::vector<float> inp(ne0 * n_t);
        for (int t = 0; t < n_t; t++)
            for (int i = 0; i < ne0; i++)
                inp[t * ne0 + i] = (float)((t+1) * 10 + i);

        // expected from first 8 experts
        std::vector<float> expected(ne1 * n_eu * n_t);
        for (int e = 0; e < n_eu; e++) {
            const float * ww = w_small.data() + e * ne0 * ne1;
            for (int r = 0; r < ne1; r++)
                expected[e * ne1 + r] = compute_dot(ww + r * ne0, inp.data(), ne0);
        }

        auto compute = [&](int ne2, const float * wdata) -> std::vector<float> {
            ggml_backend_t be = ggml_backend_init_by_name("CUDA0", nullptr);
            if (!be) return {};
            ggml_init_params ctx_p = { 32*1024*1024, nullptr, true };
            ggml_context * ctx = ggml_init(ctx_p);
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
            ggml_tensor * inp_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, 1, n_t);
            ggml_tensor * id_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_eu, n_t);
            ggml_tensor * dst = ggml_mul_mat_id(ctx, w, inp_t, id_t);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, dst);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(w, wdata, 0, ne0 * ne1 * ne2 * sizeof(float));
            ggml_backend_tensor_set(inp_t, inp.data(), 0, inp.size() * sizeof(float));
            ggml_backend_tensor_set(id_t, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ne1 * n_eu * n_t);
            ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(be);
            return out;
        };

        auto out_large = compute(ne2_large, w_large.data());
        auto out_small = compute(ne2_small, w_small.data());

        printf("\n=== ne2=256 vs ne2=%d — same weights + same ids ===\n", ne2_small);
        if (out_large.empty() || out_small.empty()) {
            printf("SKIP (CUDA not available)\n");
        } else {
            float max_diff = 0;
            int bad = 0;
            for (size_t i = 0; i < expected.size(); i++) {
                float d = fabsf(out_large[i] - out_small[i]);
                if (d > max_diff) max_diff = d;
                if (d > 1e-3f) {
                    if (bad < 5) printf("  MISMATCH[%zu]: ne2=256=%.2f ne2=%d=%.2f\n",
                        i, out_large[i], ne2_small, out_small[i]);
                    bad++;
                }
            }
            if (bad > 0) {
                printf("FAIL: %d mismatches, max_diff=%.6f\n", bad, max_diff);
                all_pass = false;
            } else {
                printf("PASS: max_diff=%.9f (ne2=256 == ne2=%d)\n", max_diff, ne2_small);
            }
        }
    }

    if (all_pass) {
        printf("\n========================================\n");
        printf("ALL TESTS PASSED — CUDA MUL_MAT_ID matches CPU\n");
        printf("========================================\n");
    } else {
        printf("\n========================================\n");
        printf("SOME TESTS FAILED — CUDA MUL_MAT_ID bug found\n");
        printf("========================================\n");
        return 1;
    }
    return 0;
}
