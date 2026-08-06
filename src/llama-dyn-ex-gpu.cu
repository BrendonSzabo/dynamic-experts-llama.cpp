#include "llama-dyn-ex.h"
#include "ggml-cuda.h"

#ifdef GGML_USE_CUDA

#include <cuda_runtime.h>

struct gpu_miss_entry { int layer; int expert_id; int target_l1_slot; };

static __device__ void copy_family(
    uint8_t * l1_dst, const uint8_t * l1_src,
    int src_slot, int dst_slot,
    size_t stride, size_t stride2) {

    GGML_UNUSED(stride2);
    if (!l1_dst || !l1_src || stride == 0) return;
    const uint8_t * src = l1_src + (size_t)src_slot * stride;
    uint8_t * dst = l1_dst + (size_t)dst_slot * stride;
    for (size_t i = 0; i < stride; i++) dst[i] = src[i];
}

__global__ void dyn_ex_slot_assign_kernel(
    const int32_t * selected_experts, int32_t * slot_ids,
    const int * expert_to_slot,
    const uint8_t * l2_gate, const uint8_t * l2_up,
    const uint8_t * l2_down, const uint8_t * l2_gate_up,
    uint8_t * l1_gate, uint8_t * l1_up,
    uint8_t * l1_down, uint8_t * l1_gate_up,
    size_t l1_str_gate, size_t l1_str_up,
    size_t l1_str_down, size_t l1_str_gate_up,
    size_t l2_str_gate, size_t l2_str_up,
    size_t l2_str_down, size_t l2_str_gate_up,
    int * l1_expert_to_slot,
    struct gpu_miss_entry * miss_buf, int * miss_count,
    int * flag,
    int * d_free_stack, int * d_stack_ptr,
    int * d_claimed_group,
    int n_expert_used, int n_tokens, int n_groups, int n_experts, int layer) {

    __shared__ int s_base;
    __shared__ bool s_ok;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int g = atomicSub(d_stack_ptr, 1) - 1;
        s_ok = (g >= 0 && g < n_groups);
        if (s_ok) s_base = g * n_expert_used;
        *d_claimed_group = g;
    }
    __syncthreads();
    if (!s_ok) return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_expert_used * n_tokens;
    if (tid >= total) return;

    int e = tid % n_expert_used;
    int t = tid / n_expert_used;
    int expert_id = selected_experts[t * n_expert_used + e];
    int l1_slot = s_base + e;

    if (expert_id < 0 || expert_id >= n_experts) {
        slot_ids[t * n_expert_used + e] = -1;
        return;
    }

    slot_ids[t * n_expert_used + e] = l1_slot;

    int l1_src = (l1_expert_to_slot && expert_id < n_experts)
        ? l1_expert_to_slot[expert_id] : -1;

    if (l1_src >= 0 && l1_src != l1_slot) {
        copy_family(l1_gate,    l1_gate,    l1_src, l1_slot, l1_str_gate,    0);
        copy_family(l1_up,      l1_up,      l1_src, l1_slot, l1_str_up,      0);
        copy_family(l1_down,    l1_down,    l1_src, l1_slot, l1_str_down,    0);
        copy_family(l1_gate_up, l1_gate_up, l1_src, l1_slot, l1_str_gate_up, 0);
        l1_expert_to_slot[expert_id] = l1_slot;
    } else {
        int ls = expert_to_slot[expert_id];
        if (ls >= 0) {
            copy_family(l1_gate,    l2_gate,    ls, l1_slot, l1_str_gate,    l2_str_gate);
            copy_family(l1_up,      l2_up,      ls, l1_slot, l1_str_up,      l2_str_up);
            copy_family(l1_down,    l2_down,    ls, l1_slot, l1_str_down,    l2_str_down);
            copy_family(l1_gate_up, l2_gate_up, ls, l1_slot, l1_str_gate_up, l2_str_gate_up);
            l1_expert_to_slot[expert_id] = l1_slot;
        } else {
            int idx = atomicAdd(miss_count, 1);
            miss_buf[idx].layer = layer;
            miss_buf[idx].expert_id = expert_id;
            miss_buf[idx].target_l1_slot = l1_slot;
        }
    }

    if (tid == 0 && *miss_count > 0) {
        *flag = 1;
        __threadfence_system();
        while (*(volatile int *)flag != 0) { __threadfence_system(); }
    }
    __syncthreads();

    __threadfence_system();
    if (*miss_count > 0) {
        int ls = *(volatile int *)&expert_to_slot[expert_id];
        if (ls >= 0) {
            copy_family(l1_gate,    l2_gate,    ls, l1_slot, l1_str_gate,    l2_str_gate);
            copy_family(l1_up,      l2_up,      ls, l1_slot, l1_str_up,      l2_str_up);
            copy_family(l1_down,    l2_down,    ls, l1_slot, l1_str_down,    l2_str_down);
            copy_family(l1_gate_up, l2_gate_up, ls, l1_slot, l1_str_gate_up, l2_str_gate_up);
            l1_expert_to_slot[expert_id] = l1_slot;
        }
    }
}

__global__ void dyn_ex_prefetch_kernel(
    const int * expert_to_slot,
    const uint8_t * l2_gate, const uint8_t * l2_up,
    const uint8_t * l2_down, const uint8_t * l2_gate_up,
    uint8_t * l1_gate, uint8_t * l1_up,
    uint8_t * l1_down, uint8_t * l1_gate_up,
    size_t l1_str_gate, size_t l1_str_up,
    size_t l1_str_down, size_t l1_str_gate_up,
    size_t l2_str_gate, size_t l2_str_up,
    size_t l2_str_down, size_t l2_str_gate_up,
    int * l1_expert_to_slot,
    const int * top_experts,
    struct gpu_miss_entry * miss_buf, int * miss_count,
    int n_prefetch, int n_expert_used, int n_l1, int n_experts, int layer) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_prefetch) return;

    int expert_id = top_experts[tid];
    if (expert_id < 0 || expert_id >= n_experts) return;

    int prefetch_slot = n_expert_used + tid;

    int l1_src = l1_expert_to_slot[expert_id];
    if (l1_src >= 0) return;

    int ls = expert_to_slot[expert_id];
    if (ls >= 0) {
        copy_family(l1_gate,    l2_gate,    ls, prefetch_slot, l1_str_gate,    l2_str_gate);
        copy_family(l1_up,      l2_up,      ls, prefetch_slot, l1_str_up,      l2_str_up);
        copy_family(l1_down,    l2_down,    ls, prefetch_slot, l1_str_down,    l2_str_down);
        copy_family(l1_gate_up, l2_gate_up, ls, prefetch_slot, l1_str_gate_up, l2_str_gate_up);
        l1_expert_to_slot[expert_id] = prefetch_slot;
    } else {
        int idx = atomicAdd(miss_count, 1);
        miss_buf[idx].layer = layer;
        miss_buf[idx].expert_id = expert_id;
        miss_buf[idx].target_l1_slot = prefetch_slot;
    }
}

__global__ void dyn_ex_release_group_kernel(
    int * d_stack, int * d_stack_ptr, int group_idx) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int pos = atomicAdd(d_stack_ptr, 1);
        d_stack[pos] = group_idx;
    }
}

static dyn_ex_cache * g_de = nullptr;

static void dyn_ex_barrier_handler(void * stream, ggml_tensor * dst) {
    if (!g_de) return;

    int il   = dst->op_params[0];
    int role = dst->op_params[1];

    if (role == 1) {
        int g = *(volatile int *)g_de->h_claimed_group;
        if (g >= 0 && g < g_de->n_groups) {
            dyn_ex_release_group_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
                g_de->d_free_stack, g_de->d_stack_ptr, g);
        }
        return;
    }

    ggml_tensor * src      = dst->src[0];
    ggml_tensor * slot_ids = dst->src[1];
    auto & l2 = g_de->l2[il];

    int n_eu = src->ne[0];
    int n_t  = src->ne[1];

    *g_de->h_miss_count = 0;
    *(volatile int *)g_de->h_sync_flag = 0;

    int total = n_eu * n_t;
    int block = 256;
    int grid  = (total + block - 1) / block;

    uint8_t * l1_g = g_de->l1_gate    ? (uint8_t *)g_de->l1_gate->data    : nullptr;
    uint8_t * l1_u = g_de->l1_up      ? (uint8_t *)g_de->l1_up->data      : nullptr;
    uint8_t * l1_d = g_de->l1_down_q4 ? (uint8_t *)g_de->l1_down_q4->data : nullptr;
    uint8_t * l1_gu= g_de->l1_gate_up ? (uint8_t *)g_de->l1_gate_up->data : nullptr;

    dyn_ex_slot_assign_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const int32_t *)src->data, (int32_t *)slot_ids->data,
        l2.d_expert_to_slot,
        l2.d_gate_data, l2.d_up_data, l2.d_down_data, l2.d_gate_up_data,
        l1_g, l1_u, l1_d, l1_gu,
        g_de->l1_stride_gate, g_de->l1_stride_up,
        g_de->l1_stride_down, g_de->l1_stride_gate_up,
        l2.gate_size, l2.up_size, l2.down_size, l2.gate_up_size,
        g_de->d_l1_expert_to_slot,
        (struct gpu_miss_entry *)g_de->d_miss_buf, g_de->d_miss_count,
        g_de->d_sync_flag,
        g_de->d_free_stack, g_de->d_stack_ptr,
        g_de->d_claimed_group,
        n_eu, n_t, g_de->n_groups, g_de->n_experts, il);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[dyn-ex] layer %d: launch error: %s\n", il, cudaGetErrorString(launch_err));
    }

    while (*(volatile int *)g_de->h_sync_flag == 0) {}

    int n = *(volatile int *)g_de->h_miss_count;
    if (n > g_de->MISS_BUF_SIZE) n = g_de->MISS_BUF_SIZE;
    for (int i = 0; i < n; i++) {
        int eid = g_de->h_miss_buf[i].expert_id;
        if (eid < 0 || eid >= g_de->reader->n_experts) continue;

        int victim = -1;
        for (int s = 0; s < g_de->n_l2; s++)
            if (l2.slot_to_expert[s] == DYN_EX_SENTINEL) { victim = s; break; }
        if (victim < 0) {
            uint64_t old = UINT64_MAX;
            for (int s = 0; s < g_de->n_l2; s++)
                if (l2.age[s] < old) { old = l2.age[s]; victim = s; }
            int ev = l2.slot_to_expert[victim];
            if (ev >= 0) l2.expert_to_slot[ev] = DYN_EX_SENTINEL;
        }
        l2.slot_to_expert[victim] = eid;
        l2.expert_to_slot[eid] = victim;
        l2.age[victim] = g_de->clock;

        auto & R = *g_de->reader;
        if (g_de->pi_gate >= 0 && l2.gate_size > 0)
            dyn_ex_read_param(&R, g_de->pi_gate, il, eid,
                l2.gate_data + (size_t)victim * l2.gate_size, l2.gate_size);
        if (g_de->pi_up >= 0 && l2.up_size > 0)
            dyn_ex_read_param(&R, g_de->pi_up, il, eid,
                l2.up_data + (size_t)victim * l2.up_size, l2.up_size);
        if (g_de->pi_down >= 0 && l2.down_size > 0)
            dyn_ex_read_param(&R, g_de->pi_down, il, eid,
                l2.down_data + (size_t)victim * l2.down_size, l2.down_size);
        if (g_de->pi_gate_up >= 0 && l2.gate_up_size > 0)
            dyn_ex_read_param(&R, g_de->pi_gate_up, il, eid,
                l2.gate_up_data + (size_t)victim * l2.gate_up_size, l2.gate_up_size);
    }
    *(volatile int *)g_de->h_miss_count = 0;
    *(volatile int *)g_de->h_sync_flag = 0;

    cudaDeviceSynchronize();

    std::vector<int> ids(n_eu);
    cudaMemcpy(ids.data(), src->data, n_eu * sizeof(int), cudaMemcpyDeviceToHost);
    if (il < (int)g_de->freq.size()) {
        auto & f = g_de->freq[il];
        for (int eid : ids) if (eid >= 0 && eid < (int)f.size()) f[eid]++;
    }

    int next_il = il + 1;
    if (next_il < (int)g_de->freq.size() && !g_de->freq[next_il].empty()) {
        int n_prefetch = g_de->n_l1 - n_eu;
        if (n_prefetch > 0) {
            std::vector<std::pair<int,int>> ranked;
            auto & fn = g_de->freq[next_il];
            for (int eid = 0; eid < (int)fn.size(); eid++)
                if (fn[eid] > 0) ranked.push_back({fn[eid], eid});
            std::sort(ranked.rbegin(), ranked.rend());
            int p_slot = n_eu;
            for (int k = 0; k < (int)ranked.size() && p_slot < g_de->n_l1; k++) {
                int eid = ranked[k].second;
                if (g_de->h_l1_expert_to_slot[eid] >= 0) continue;
                auto & l2n = g_de->l2[next_il];
                int ls = l2n.expert_to_slot[eid];
                auto & R = *g_de->reader;
                if (ls >= 0 && l2n.gate_data && l2n.gate_size > 0) {
                    size_t sz = g_de->l1_stride_gate;
                    if (sz > 0) cudaMemcpy(
                        (char *)g_de->l1_gate->data + (size_t)p_slot * sz,
                        l2n.gate_data + (size_t)ls * l2n.gate_size, sz, cudaMemcpyHostToDevice);
                    if (g_de->pi_up >= 0 && g_de->l1_up && l2n.up_data && l2n.up_size > 0)
                        cudaMemcpy((char *)g_de->l1_up->data + (size_t)p_slot * g_de->l1_stride_up,
                            l2n.up_data + (size_t)ls * l2n.up_size, g_de->l1_stride_up, cudaMemcpyHostToDevice);
                    if (g_de->pi_down >= 0 && g_de->l1_down_q4 && l2n.down_data && l2n.down_size > 0)
                        cudaMemcpy((char *)g_de->l1_down_q4->data + (size_t)p_slot * g_de->l1_stride_down,
                            l2n.down_data + (size_t)ls * l2n.down_size, g_de->l1_stride_down, cudaMemcpyHostToDevice);
                    if (g_de->pi_gate_up >= 0 && g_de->l1_gate_up && l2n.gate_up_data && l2n.gate_up_size > 0)
                        cudaMemcpy((char *)g_de->l1_gate_up->data + (size_t)p_slot * g_de->l1_stride_gate_up,
                            l2n.gate_up_data + (size_t)ls * l2n.gate_up_size, g_de->l1_stride_gate_up, cudaMemcpyHostToDevice);
                    g_de->h_l1_expert_to_slot[eid] = p_slot;
                    p_slot++;
                }
            }
        }
    }
}

void dyn_ex_register_gpu_handler(dyn_ex_cache * de) {
    g_de = de;
    ggml_cuda_set_dyn_ex_barrier(dyn_ex_barrier_handler);
}

#endif
