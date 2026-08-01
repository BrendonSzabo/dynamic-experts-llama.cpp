// dyn-ex GPU dequantize: Q4_K/Q6_K → FP16

static __global__ void dyn_ex_dequant_q4_k_f16(
    const block_q4_K * __restrict__ x, ggml_fp16_t * __restrict__ y, int n_blocks) {
    int ib = blockIdx.x;
    if (ib >= n_blocks) return;
    const block_q4_K & b = x[ib];
    float d = GGML_FP16_TO_FP32(b.d);
    float dmin = GGML_FP16_TO_FP32(b.dmin);
    int base = ib * QK_K;
    for (int j = threadIdx.x; j < QK_K; j += blockDim.x) {
        int q = (b.qs[j/2] >> (4*(j%2))) & 0xF;
        float v = d * q - dmin;
        y[base + j] = GGML_FP32_TO_FP16(v);
    }
}

static __global__ void dyn_ex_dequant_q6_k_f16(
    const block_q6_K * __restrict__ x, ggml_fp16_t * __restrict__ y, int n_blocks) {
    int ib = blockIdx.x;
    if (ib >= n_blocks) return;
    const block_q6_K & b = x[ib];
    float d = GGML_FP16_TO_FP32(b.d);
    int base = ib * QK_K;
    for (int j = threadIdx.x; j < QK_K; j += blockDim.x) {
        int q = ((b.ql[j/2] >> (4*(j%2))) & 0xF) | ((b.qh[j/4] >> (2*(j%4))) & 0x3) << 4;
        float v = d * q - 32.0f * d;
        y[base + j] = GGML_FP32_TO_FP16(v);
    }
}
