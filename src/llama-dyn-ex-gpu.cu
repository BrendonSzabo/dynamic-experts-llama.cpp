#include "llama-dyn-ex.h"

#ifdef GGML_USE_CUDA

#include <cuda_runtime.h>

struct gpu_miss_entry { int layer; int expert_id; };

__device__ static void copy_expert_family(
    uint8_t * __restrict__ l1, const uint8_t * __restrict__ l2,
    int l2_slot, int l1_slot,
    size_t l1_stride, size_t l2_row) {

    if (!l1 || !l2 || l1_stride == 0 || l2_row == 0) return;
    const uint8_t * src = l2 + (size_t)l2_slot * l2_row;
    uint8_t * dst = l1 + (size_t)l1_slot * l1_stride;
    for (size_t i = 0; i < l1_stride; i++) dst[i] = src[i];
}

__global__ void dyn_ex_slot_assign_kernel(
    const int32_t * __restrict__ selected_experts,
    int32_t * __restrict__ slot_ids,
    const int * __restrict__ expert_to_slot,
    const uint8_t * __restrict__ l2_gate,
    const uint8_t * __restrict__ l2_up,
    const uint8_t * __restrict__ l2_down,
    const uint8_t * __restrict__ l2_gate_up,
    uint8_t * __restrict__ l1_gate,
    uint8_t * __restrict__ l1_up,
    uint8_t * __restrict__ l1_down,
    uint8_t * __restrict__ l1_gate_up,
    size_t l1_stride_gate,
    size_t l1_stride_up,
    size_t l1_stride_down,
    size_t l1_stride_gate_up,
    size_t l2_row_gate,
    size_t l2_row_up,
    size_t l2_row_down,
    size_t l2_row_gate_up,
    struct gpu_miss_entry * miss_buf,
    int * miss_count,
    int * d_free_stack,
    int * d_stack_ptr,
    volatile int * sync_flag,
    volatile int * misses_posted,
    int n_expert_used,
    int n_tokens,
    int n_groups,
    int n_experts,
    int layer) {

    __shared__ int s_base_slot;
    __shared__ bool s_ok;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int g = atomicSub(d_stack_ptr, 1) - 1;
        if (g < 0 || g >= n_groups) {
            s_ok = false;
        } else {
            s_base_slot = g * n_expert_used;
            s_ok = true;
        }
    }
    __syncthreads();
    if (!s_ok) return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_expert_used * n_tokens;
    if (tid >= total) return;

    int e = tid % n_expert_used;
    int t = tid / n_expert_used;
    int expert_id = selected_experts[t * n_expert_used + e];
    int l1_slot = s_base_slot + e;

    if (expert_id < 0 || expert_id >= n_experts) {
        slot_ids[t * n_expert_used + e] = -1;
        return;
    }

    int l2_slot = expert_to_slot[expert_id];
    if (l2_slot >= 0) {
        copy_expert_family(l1_gate,    l2_gate,    l2_slot, l1_slot, l1_stride_gate,    l2_row_gate);
        copy_expert_family(l1_up,      l2_up,      l2_slot, l1_slot, l1_stride_up,      l2_row_up);
        copy_expert_family(l1_down,    l2_down,    l2_slot, l1_slot, l1_stride_down,    l2_row_down);
        copy_expert_family(l1_gate_up, l2_gate_up, l2_slot, l1_slot, l1_stride_gate_up, l2_row_gate_up);
    } else {
        int idx = atomicAdd(miss_count, 1);
        miss_buf[idx].layer = layer;
        miss_buf[idx].expert_id = expert_id;
    }

    slot_ids[t * n_expert_used + e] = l1_slot;

    if (tid == 0) {
        *misses_posted = 1;
        __threadfence_system();
        if (*miss_count > 0) {
            while (*sync_flag == 0) { __threadfence_system(); }
        }
    }
    __syncthreads();

    if (*miss_count > 0) {
        l2_slot = expert_to_slot[expert_id];
        if (l2_slot >= 0) {
            copy_expert_family(l1_gate,    l2_gate,    l2_slot, l1_slot, l1_stride_gate,    l2_row_gate);
            copy_expert_family(l1_up,      l2_up,      l2_slot, l1_slot, l1_stride_up,      l2_row_up);
            copy_expert_family(l1_down,    l2_down,    l2_slot, l1_slot, l1_stride_down,    l2_row_down);
            copy_expert_family(l1_gate_up, l2_gate_up, l2_slot, l1_slot, l1_stride_gate_up, l2_row_gate_up);
        }
    }
}

__global__ void dyn_ex_release_group_kernel(int * d_stack, int * d_stack_ptr, int group_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int pos = atomicAdd(d_stack_ptr, 1);
        d_stack[pos] = group_idx;
    }
}

void dyn_ex_cache::launch_slot_assign(
    const int32_t * d_selected_experts,
    int32_t * d_slot_ids,
    const int * d_l2_expert_to_slot,
    const uint8_t * d_l2_gate_data,
    const uint8_t * d_l2_up_data,
    const uint8_t * d_l2_down_data,
    const uint8_t * d_l2_gate_up_data,
    uint8_t * d_l1_gate,
    uint8_t * d_l1_up,
    uint8_t * d_l1_down,
    uint8_t * d_l1_gate_up,
    size_t l1_stride_gate,
    size_t l1_stride_up,
    size_t l1_stride_down,
    size_t l1_stride_gate_up,
    size_t l2_gate_row,
    size_t l2_up_row,
    size_t l2_down_row,
    size_t l2_gate_up_row,
    int n_expert_used,
    int n_tokens,
    int n_groups,
    int base_slot,
    int n_experts,
    int layer,
    void * stream) {

    GGML_UNUSED(base_slot);

    int total = n_expert_used * n_tokens;
    int block = 256;
    int grid = (total + block - 1) / block;

    dyn_ex_slot_assign_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_selected_experts, d_slot_ids,
        d_l2_expert_to_slot,
        d_l2_gate_data, d_l2_up_data, d_l2_down_data, d_l2_gate_up_data,
        d_l1_gate, d_l1_up, d_l1_down, d_l1_gate_up,
        l1_stride_gate, l1_stride_up, l1_stride_down, l1_stride_gate_up,
        l2_gate_row, l2_up_row, l2_down_row, l2_gate_up_row,
        (struct gpu_miss_entry *)d_miss_buf, d_miss_count,
        d_free_stack, d_stack_ptr,
        (volatile int *)d_sync_flag, (volatile int *)d_misses_posted,
        n_expert_used, n_tokens, n_groups, n_experts, layer);
}

void dyn_ex_cache::release_group(int * d_stack, int * d_stack_ptr, int group_idx, void * stream) {
    dyn_ex_release_group_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(d_stack, d_stack_ptr, group_idx);
}

#endif // GGML_USE_CUDA
