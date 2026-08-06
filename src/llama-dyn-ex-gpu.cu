#include "llama-dyn-ex.h"
#include "ggml-cuda.h"

#ifdef GGML_USE_CUDA

#include <cuda_runtime.h>

struct gpu_miss_entry { int layer; int expert_id; };

static __device__ void copy_family(
    uint8_t * l1, const uint8_t * l2,
    int l2_slot, int l1_slot,
    size_t l1_stride, size_t l2_row) {

    if (!l1 || !l2 || l1_stride == 0 || l2_row == 0) return;
    const uint8_t * src = l2 + (size_t)l2_slot * l2_row;
    uint8_t * dst = l1 + (size_t)l1_slot * l1_stride;
    for (size_t i = 0; i < l1_stride; i++) dst[i] = src[i];
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
    size_t l2_row_gate, size_t l2_row_up,
    size_t l2_row_down, size_t l2_row_gate_up,
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

    int l2_slot = expert_to_slot[expert_id];
    if (l2_slot >= 0) {
        copy_family(l1_gate,    l2_gate,    l2_slot, l1_slot, l1_str_gate,    l2_row_gate);
        copy_family(l1_up,      l2_up,      l2_slot, l1_slot, l1_str_up,      l2_row_up);
        copy_family(l1_down,    l2_down,    l2_slot, l1_slot, l1_str_down,    l2_row_down);
        copy_family(l1_gate_up, l2_gate_up, l2_slot, l1_slot, l1_str_gate_up, l2_row_gate_up);
    } else {
        int idx = atomicAdd(miss_count, 1);
        miss_buf[idx].layer = layer;
        miss_buf[idx].expert_id = expert_id;
    }

    if (tid == 0 && *miss_count > 0) {
        *flag = 1;
        __threadfence_system();
        while (*(volatile int *)flag != 0) { __threadfence_system(); }
    }
    __syncthreads();

    if (*miss_count > 0) {
        int ls = expert_to_slot[expert_id];
        if (ls >= 0) {
            copy_family(l1_gate,    l2_gate,    ls, l1_slot, l1_str_gate,    l2_row_gate);
            copy_family(l1_up,      l2_up,      ls, l1_slot, l1_str_up,      l2_row_up);
            copy_family(l1_down,    l2_down,    ls, l1_slot, l1_str_down,    l2_row_down);
            copy_family(l1_gate_up, l2_gate_up, ls, l1_slot, l1_str_gate_up, l2_row_gate_up);
        }
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
    fprintf(stderr, "[dyn-ex] layer=%d role=%d stream=%p\n", il, role, stream);

    if (role == 1) {
        int g = *(volatile int *)g_de->h_claimed_group;
        fprintf(stderr, "[dyn-ex] layer %d: release group=%d\n", il, g);
        if (g >= 0 && g < g_de->n_groups) {
            dyn_ex_release_group_kernel<<<1, 1>>>(
                g_de->d_free_stack, g_de->d_stack_ptr, g);
            cudaError_t rerr = cudaGetLastError();
            if (rerr != cudaSuccess) fprintf(stderr, "[dyn-ex] layer %d: release kernel error: %s\n", il, cudaGetErrorString(rerr));
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

    dyn_ex_slot_assign_kernel<<<grid, block>>>(
        (const int32_t *)src->data, (int32_t *)slot_ids->data,
        l2.d_expert_to_slot,
        l2.d_gate_data, l2.d_up_data, l2.d_down_data, l2.d_gate_up_data,
        l1_g, l1_u, l1_d, l1_gu,
        g_de->l1_stride_gate, g_de->l1_stride_up,
        g_de->l1_stride_down, g_de->l1_stride_gate_up,
        l2.gate_row, l2.up_row, l2.down_row, l2.gate_up_row,
        (struct gpu_miss_entry *)g_de->d_miss_buf, g_de->d_miss_count,
        g_de->d_sync_flag,
        g_de->d_free_stack, g_de->d_stack_ptr,
        g_de->d_claimed_group,
        n_eu, n_t, g_de->n_groups, g_de->n_experts, il);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[dyn-ex] layer %d: launch error: %s\n", il, cudaGetErrorString(launch_err));
    }

    fprintf(stderr, "[dyn-ex] layer %d: kernel launched, waiting for flag...\n", il);
    while (*(volatile int *)g_de->h_sync_flag == 0) {}
    fprintf(stderr, "[dyn-ex] layer %d: flag set, miss_count=%d\n", il, *(volatile int *)g_de->h_miss_count);

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
                l2.gate_data + (size_t)victim * l2.gate_row, l2.gate_size);
        if (g_de->pi_up >= 0 && l2.up_size > 0)
            dyn_ex_read_param(&R, g_de->pi_up, il, eid,
                l2.up_data + (size_t)victim * l2.up_row, l2.up_size);
        if (g_de->pi_down >= 0 && l2.down_size > 0)
            dyn_ex_read_param(&R, g_de->pi_down, il, eid,
                l2.down_data + (size_t)victim * l2.down_row, l2.down_size);
        if (g_de->pi_gate_up >= 0 && l2.gate_up_size > 0)
            dyn_ex_read_param(&R, g_de->pi_gate_up, il, eid,
                l2.gate_up_data + (size_t)victim * l2.gate_up_row, l2.gate_up_size);
    }
    *(volatile int *)g_de->h_miss_count = 0;
    *(volatile int *)g_de->h_sync_flag = 0;

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[dyn-ex] layer %d: sync error BEFORE check: %s\n", il, cudaGetErrorString(sync_err));
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[dyn-ex] CUDA error layer=%d: %s\n", il, cudaGetErrorString(err));
    }
}

void dyn_ex_register_gpu_handler(dyn_ex_cache * de) {
    g_de = de;
    ggml_cuda_set_dyn_ex_barrier(dyn_ex_barrier_handler);
}

#endif
