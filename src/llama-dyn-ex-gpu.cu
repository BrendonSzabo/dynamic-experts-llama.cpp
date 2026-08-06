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
    if (stride > (1ULL << 30)) { printf("[dyn-ex] FATAL stride=%zu\n", stride); asm("trap;"); }
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
        if (s_ok) s_base = g * n_expert_used;
        *d_claimed_group = g;
        __threadfence();
        *d_group_ok = s_ok ? 1 : 0;
    }
    if (blockIdx.x > 0 || threadIdx.x > 0)
        while (*(volatile int *)d_group_ok == -1) {}
    __threadfence();
    s_ok = (*d_group_ok == 1);
    s_base = s_ok ? *d_group_base : 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_expert_used * n_tokens;
    if (tid >= total) return;
    int e = tid % n_expert_used;
    int t = tid / n_expert_used;
    if (!s_ok) { slot_ids[t * n_expert_used + e] = -1; return; }

    int expert_id = selected_experts[t * n_expert_used + e];
    int l1_slot = s_base + e;
    if (expert_id < 0 || expert_id >= n_experts) {
        slot_ids[t * n_expert_used + e] = -1; return;
    }
    slot_ids[t * n_expert_used + e] = l1_slot;

    if (tid == 0) printf("[dyn-ex] L%d launch: p=%p l2g=%p l2u=%p l2d=%p l2gu=%p sg=%lu su=%lu sd=%lu sgu=%lu\n",
        layer, p, (void*)l2_gate, (void*)l2_up, (void*)l2_down, (void*)l2_gate_up,
        (unsigned long)l2_str_gate, (unsigned long)l2_str_up, (unsigned long)l2_str_down, (unsigned long)l2_str_gate_up);

    int l1_src = l1_expert_to_slot ? l1_expert_to_slot[expert_id] : -1;
    uint8_t * l1_dn   = p->l1_down[0] ? p->l1_down[0] : p->l1_down[1];
    size_t l1_str_dn = p->l1_down[0] ? p->l1_str_down[0] : p->l1_str_down[1];

    if (l1_src >= 0 && l1_src != l1_slot) {
        copy_family(p->l1_gate,    p->l1_gate,    l1_src, l1_slot, p->l1_str_gate,    0);
        copy_family(p->l1_up,      p->l1_up,      l1_src, l1_slot, p->l1_str_up,      0);
        copy_family(l1_dn,         l1_dn,         l1_src, l1_slot, l1_str_dn,          0);
        copy_family(p->l1_gate_up, p->l1_gate_up, l1_src, l1_slot, p->l1_str_gate_up, 0);
        l1_expert_to_slot[expert_id] = l1_slot;
        if (l1_slot_to_expert) {
            l1_slot_to_expert[l1_src]  = -1;
            l1_slot_to_expert[l1_slot] = expert_id;
        }
    } else {
        int ls = expert_to_slot[expert_id];
        printf("[dyn-ex] L%d t%d: eid=%d ls=%d slot=%d\n", layer, tid, expert_id, ls, l1_slot);
        if (ls >= 0) {
            copy_family(p->l1_gate,    l2_gate,       ls, l1_slot, p->l1_str_gate,    l2_str_gate);
            copy_family(p->l1_up,      l2_up,         ls, l1_slot, p->l1_str_up,      l2_str_up);
            copy_family(l1_dn,         l2_down,       ls, l1_slot, l1_str_dn,         l2_str_down);
            copy_family(p->l1_gate_up, l2_gate_up,    ls, l1_slot, p->l1_str_gate_up, l2_str_gate_up);
            l1_expert_to_slot[expert_id] = l1_slot;
            if (l1_slot_to_expert) l1_slot_to_expert[l1_slot] = expert_id;
        } else {
            printf("[dyn-ex] L%d t%d: MISS eid=%d\n", layer, tid, expert_id);
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
        printf("[dyn-ex] L%d t%d RETRY: eid=%d ls=%d\n", layer, tid, expert_id, ls);
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
    } else {
        int idx = atomicAdd(miss_count, 1);
        if (idx < dyn_ex_cache::MISS_BUF_SIZE) {
            miss_buf[idx].layer          = layer;
            miss_buf[idx].expert_id      = expert_id;
            miss_buf[idx].target_l1_slot = prefetch_slot;
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

static void handle_misses(dyn_ex_cache * de, int n, int il) {
    if (n > de->MISS_BUF_SIZE) n = de->MISS_BUF_SIZE;
    auto & l2 = de->l2[il];
    for (int i = 0; i < n; i++) {
        int eid = de->h_miss_buf[i].expert_id;
        if (eid < 0 || eid >= de->reader->n_experts) continue;
        int victim = -1;
        for (int s = 0; s < de->n_l2; s++)
            if (l2.slot_to_expert[s] == DYN_EX_SENTINEL) { victim = s; break; }
        if (victim < 0) {
            uint64_t old = UINT64_MAX;
            for (int s = 0; s < de->n_l2; s++)
                if (l2.age[s] < old) { old = l2.age[s]; victim = s; }
            int ev = l2.slot_to_expert[victim];
            if (ev >= 0) l2.expert_to_slot[ev] = DYN_EX_SENTINEL;
        }
        l2.slot_to_expert[victim] = eid;
        l2.expert_to_slot[eid] = victim;
        l2.age[victim] = de->clock;
        auto & R = *de->reader;
        if (de->pi_gate >= 0 && l2.gate_size > 0)
            dyn_ex_read_param(&R, de->pi_gate, il, eid,
                l2.gate_data + (size_t)victim * l2.gate_row, l2.gate_size);
        if (de->pi_up >= 0 && l2.up_size > 0)
            dyn_ex_read_param(&R, de->pi_up, il, eid,
                l2.up_data + (size_t)victim * l2.up_row, l2.up_size);
        if (de->pi_down >= 0 && l2.down_size > 0)
            dyn_ex_read_param(&R, de->pi_down, il, eid,
                l2.down_data + (size_t)victim * l2.down_row, l2.down_size);
        if (de->pi_gate_up >= 0 && l2.gate_up_size > 0)
            dyn_ex_read_param(&R, de->pi_gate_up, il, eid,
                l2.gate_up_data + (size_t)victim * l2.gate_up_row, l2.gate_up_size);
    }
    *(volatile int *)de->h_miss_count = 0;
    *(volatile int *)de->h_sync_flag = 0;
}

static void dyn_ex_watchdog_loop(dyn_ex_cache * de) {
    while (!de->watchdog_stop.load(std::memory_order_relaxed)) {
        if (*(volatile int *)de->h_sync_flag != 0) {
            int layer = de->h_miss_buf[0].layer;
            int n = *(volatile int *)de->h_miss_count;
            handle_misses(de, n, layer);
        }
        std::this_thread::yield();
    }
}

void dyn_ex_watchdog_start(dyn_ex_cache * de) {
    if (!de || de->watchdog_thread.joinable()) return;
    de->watchdog_stop.store(false, std::memory_order_relaxed);
    de->watchdog_thread = std::thread(dyn_ex_watchdog_loop, de);
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
