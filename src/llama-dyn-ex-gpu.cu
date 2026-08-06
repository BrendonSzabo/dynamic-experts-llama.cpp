#include "llama-dyn-ex.h"
#include "ggml-cuda.h"

#ifdef GGML_USE_CUDA

#include <cuda_runtime.h>

struct gpu_miss_entry { int layer; int expert_id; int target_l1_slot; };

struct gpu_dyn_ex_params {
    uint8_t * l1_gate;
    uint8_t * l1_up;
    uint8_t * l1_down[4];
    uint8_t * l1_gate_up;
    size_t l1_str_gate;
    size_t l1_str_up;
    size_t l1_str_down[4];
    size_t l1_str_gate_up;
    int    n_down_families;
};
static gpu_dyn_ex_params   d_params_host;
static gpu_dyn_ex_params * d_params_dev = nullptr;

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
    const struct gpu_dyn_ex_params * p,
    const int32_t * selected_experts, int32_t * slot_ids,
    const int * expert_to_slot,
    const uint8_t * l2_gate, const uint8_t * l2_up,
    const uint8_t * l2_down, const uint8_t * l2_gate_up,
    size_t l2_str_gate, size_t l2_str_up,
    size_t l2_str_down, size_t l2_str_gate_up,
    int * l1_expert_to_slot,
    int * l1_slot_to_expert,
    struct gpu_miss_entry * miss_buf, int * miss_count,
    int * flag,
    int * d_free_stack, int * d_stack_ptr,
    int * d_claimed_group,
    int * d_group_ok, int * d_group_base,
    int n_expert_used, int n_tokens, int n_groups, int n_experts, int layer) {

    int s_base = 0;
    bool s_ok = false;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int g = atomicSub(d_stack_ptr, 1) - 1;
        s_ok = (g >= 0 && g < n_groups);
        if (s_ok) {
            s_base = g * n_expert_used;
            *d_group_base = s_base;
            int n_l1 = n_groups * n_expert_used;
            int slot_end = s_base + n_expert_used;
            if (l1_slot_to_expert) {
                for (int s = s_base; s < slot_end; s++) {
                    int eid = l1_slot_to_expert[s];
                    if (eid == DYN_EX_SENTINEL) continue;
                    if (l1_expert_to_slot) l1_expert_to_slot[eid] = DYN_EX_SENTINEL;
                    l1_slot_to_expert[s] = DYN_EX_SENTINEL;
                }
            }
        }
        *d_claimed_group = g;
        __threadfence();
        *d_group_ok = s_ok ? 1 : 0;
    }

    if (blockIdx.x > 0 || threadIdx.x > 0)
        while (*(volatile int *)d_group_ok == -1) {}
    __threadfence();

    s_ok   = (*d_group_ok == 1);
    s_base = s_ok ? *d_group_base : 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_expert_used * n_tokens;
    if (tid >= total) return;

    int e = tid % n_expert_used;
    int t = tid / n_expert_used;

    if (!s_ok) {
        slot_ids[t * n_expert_used + e] = -1;
        return;
    }

    int expert_id = selected_experts[t * n_expert_used + e];
    int l1_slot = s_base + e;

    if (expert_id < 0 || expert_id >= n_experts) {
        slot_ids[t * n_expert_used + e] = -1;
        return;
    }

    slot_ids[t * n_expert_used + e] = l1_slot;

    int n_l1 = n_groups * n_expert_used;
    int l1_src = (l1_expert_to_slot && expert_id < n_experts)
        ? l1_expert_to_slot[expert_id] : -1;
    if (l1_src >= n_l1) l1_src = -1;

    uint8_t * l1_dn   = p->l1_down[0] ? p->l1_down[0] : p->l1_down[1];
    size_t l1_str_dn = p->l1_down[0] ? p->l1_str_down[0] : p->l1_str_down[1];

    if (l1_src >= 0 && l1_src != l1_slot) {
        copy_family(p->l1_gate,    p->l1_gate,    l1_src, l1_slot, p->l1_str_gate,    0);
        copy_family(p->l1_up,      p->l1_up,      l1_src, l1_slot, p->l1_str_up,      0);
        copy_family(l1_dn,         l1_dn,         l1_src, l1_slot, l1_str_dn,          0);
        copy_family(p->l1_gate_up, p->l1_gate_up, l1_src, l1_slot, p->l1_str_gate_up, 0);
        l1_expert_to_slot[expert_id] = l1_slot;
        if (l1_slot_to_expert) {
            l1_slot_to_expert[l1_src]  = DYN_EX_SENTINEL;
            l1_slot_to_expert[l1_slot] = expert_id;
        }
    } else {
        int ls = expert_to_slot[expert_id];
        if (ls >= 0) {
            copy_family(p->l1_gate,    l2_gate,       ls, l1_slot, p->l1_str_gate,    l2_str_gate);
            copy_family(p->l1_up,      l2_up,         ls, l1_slot, p->l1_str_up,      l2_str_up);
            copy_family(l1_dn,         l2_down,       ls, l1_slot, l1_str_dn,         l2_str_down);
            copy_family(p->l1_gate_up, l2_gate_up,    ls, l1_slot, p->l1_str_gate_up, l2_str_gate_up);
            l1_expert_to_slot[expert_id] = l1_slot;
            if (l1_slot_to_expert) l1_slot_to_expert[l1_slot] = expert_id;
        } else {
            int idx = atomicAdd(miss_count, 1);
            if (idx < dyn_ex_cache::MISS_BUF_SIZE) {
                miss_buf[idx].layer          = layer;
                miss_buf[idx].expert_id      = expert_id;
                miss_buf[idx].target_l1_slot = l1_slot;
            }
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
            copy_family(p->l1_gate,    l2_gate,       ls, l1_slot, p->l1_str_gate,    l2_str_gate);
            copy_family(p->l1_up,      l2_up,         ls, l1_slot, p->l1_str_up,      l2_str_up);
            copy_family(l1_dn,         l2_down,       ls, l1_slot, l1_str_dn,         l2_str_down);
            copy_family(p->l1_gate_up, l2_gate_up,    ls, l1_slot, p->l1_str_gate_up, l2_str_gate_up);
            l1_expert_to_slot[expert_id] = l1_slot;
            if (l1_slot_to_expert) l1_slot_to_expert[l1_slot] = expert_id;
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

void dyn_ex_barrier_run(dyn_ex_cache * de, void * stream, ggml_tensor * dst) {
    if (!de) return;

    int il   = dst->op_params[0];
    int role = dst->op_params[1];

    if (role == 1) {
        int g = *(volatile int *)de->h_claimed_group;
        if (g >= 0 && g < de->n_groups)
            dyn_ex_release_group_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
                de->d_free_stack, de->d_stack_ptr, g);
        return;
    }
    if (il < 0 || il >= (int)de->l2.size()) return;

    ggml_tensor * src      = dst->src[0];
    ggml_tensor * slot_ids = dst->src[1];
    int n_eu = src->ne[0], n_t = src->ne[1];
    if (n_eu <= 0 || n_t <= 0) return;

    de->clock++;
    *(volatile int *)de->h_group_ok = -1;

    if (!d_params_dev) {
        cudaMalloc(&d_params_dev, sizeof(gpu_dyn_ex_params));
        d_params_host.l1_gate    = de->l1_gate    ? (uint8_t *)de->l1_gate->data    : nullptr;
        d_params_host.l1_up      = de->l1_up      ? (uint8_t *)de->l1_up->data      : nullptr;
        d_params_host.l1_gate_up = de->l1_gate_up ? (uint8_t *)de->l1_gate_up->data : nullptr;
        for (int d = 0; d < de->n_down_families; d++)
            d_params_host.l1_down[d] = de->l1_down[d] ? (uint8_t *)de->l1_down[d]->data : nullptr;
        d_params_host.l1_str_gate    = de->l1_stride_gate;
        d_params_host.l1_str_up      = de->l1_stride_up;
        d_params_host.l1_str_gate_up = de->l1_stride_gate_up;
        for (int d = 0; d < de->n_down_families; d++)
            d_params_host.l1_str_down[d] = de->l1_stride_down_arr[d];
        d_params_host.n_down_families = de->n_down_families;
        cudaMemcpy(d_params_dev, &d_params_host, sizeof(gpu_dyn_ex_params), cudaMemcpyHostToDevice);
    }

    auto & l2 = de->l2[il];
    int total = n_eu * n_t, block = 256, grid = (total + block - 1) / block;
    if (grid > 65535) grid = 1;

    fprintf(stderr, "[dyn-ex] L%d str: g=%lu u=%lu d=%lu gu=%lu l2sg=%lu l2su=%lu l2sd=%lu\n",
        il, (unsigned long)de->l1_stride_gate, (unsigned long)de->l1_stride_up,
        (unsigned long)de->l1_stride_down, (unsigned long)de->l1_stride_gate_up,
        (unsigned long)l2.gate_size, (unsigned long)l2.up_size, (unsigned long)l2.down_size);

    dyn_ex_slot_assign_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_params_dev,
        (const int32_t *)src->data, (int32_t *)slot_ids->data,
        l2.d_expert_to_slot,
        l2.d_gate_data, l2.d_up_data, l2.d_down_data, l2.d_gate_up_data,
        l2.gate_size, l2.up_size, l2.down_size, l2.gate_up_size,
        de->d_l1_expert_to_slot, de->d_l1_slot_to_expert,
        (struct gpu_miss_entry *)de->d_miss_buf, de->d_miss_count,
        de->d_sync_flag,
        de->d_free_stack, de->d_stack_ptr,
        de->d_claimed_group,
        de->d_group_ok, de->d_group_base,
        n_eu, n_t, de->n_groups, de->n_experts, il);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        fprintf(stderr, "[dyn-ex] L%d launch error: %s\n", il, cudaGetErrorString(launch_err));
}

static void dyn_ex_barrier_handler(void * stream, ggml_tensor * dst) {
    dyn_ex_barrier_run(g_de, stream, dst);
}

void dyn_ex_register_gpu_handler(dyn_ex_cache * de) {
    g_de = de;
    ggml_cuda_set_dyn_ex_barrier(dyn_ex_barrier_handler);
}

#endif
