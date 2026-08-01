#!/usr/bin/env python3
"""
Convert GGUF model expert weights to the llm_arch_expert binary format.
With --fp16: dequantize Q4_K/Q6_K to FP16 in the output.

Usage:
    python convert-gguf-to-expert-binary.py model.gguf expert_weights.bin
    python convert-gguf-to-expert-binary.py model.gguf expert_weights_fp16.bin --fp16
"""

import sys, struct, argparse
from collections import defaultdict, Counter
import numpy as np
from gguf import GGUFReader

MAGIC = b"VLLM\x02"
VERSION = 2
HEADER_SIZE = 16384
PAGE_SIZE = 4096

DTYPE_CODE_F16 = 1
DTYPE_CODE_F16 = 2
DTYPE_CODE_Q4_K = 32
DTYPE_CODE_Q6_K = 34

QK_K = 256

def dequant_q4_k(block_bytes):
    """Dequantize one Q4_K super-block (256 elements) to float32."""
    d = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(block_bytes[2:4], dtype=np.float16)[0].astype(np.float32)
    scales = np.frombuffer(block_bytes[4:16], dtype=np.uint8).astype(np.float32)
    qs = np.frombuffer(block_bytes[16:144], dtype=np.uint8)
    result = np.zeros(QK_K, dtype=np.float32)
    for j in range(QK_K):
        q = (qs[j//2] >> (4*(j%2))) & 0xF
        result[j] = d * scales[j//32] - dmin
    return result

def dequant_q6_k(block_bytes):
    """Dequantize one Q6_K super-block (256 elements) to float32."""
    d = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(np.float32)
    ql = np.frombuffer(block_bytes[2:130], dtype=np.uint8)
    qh = np.frombuffer(block_bytes[130:194], dtype=np.uint8)
    result = np.zeros(QK_K, dtype=np.float32)
    for j in range(QK_K):
        q = ((ql[j//2] >> (4*(j%2))) & 0xF) | ((qh[j//4] >> (2*(j%4))) & 0x3) << 4
        result[j] = d * q - 32.0 * d
    return result

def dequant_to_fp16(raw_bytes, ggml_type, shape):
    """Dequantize Q4_K or Q6_K tensor to FP16."""
    ne0, ne1 = shape[0], shape[1]
    n_blocks = (ne0 * ne1) // QK_K
    if ggml_type == 12:
        block_size, dequant_fn = 144, dequant_q4_k
    elif ggml_type == 14:
        block_size, dequant_fn = 210, dequant_q6_k
    else:
        raise ValueError(f"Unsupported type {ggml_type}")
    result = np.zeros(ne0 * ne1, dtype=np.float16)
    for b in range(n_blocks):
        block = raw_bytes[b*block_size:(b+1)*block_size]
        f32_block = dequant_fn(block)
        np.clip(f32_block, -65504.0, 65504.0, out=f32_block)
        result[b*QK_K:(b+1)*QK_K] = f32_block.astype(np.float16)
    return result.tobytes()

def round_up(n, boundary=PAGE_SIZE):
    return ((n + boundary - 1) // boundary) * boundary

def find_expert_params(reader):
    layer_data = defaultdict(dict)
    expert_counts = []
    for tensor in reader.tensors:
        name = tensor.name
        shape = list(tensor.shape)
        if len(shape) < 2:
            continue
        n_exp_dim = shape[-1]
        if n_exp_dim < 4 or n_exp_dim > 4096:
            continue
        layer = -1
        for part in name.split("."):
            try:
                layer = int(part); break
            except ValueError: pass
        if layer < 0: continue
        if "gate_up_exps" in name: param = "gate_up_proj"
        elif "gate_exps" in name: param = "gate_proj"
        elif "up_exps" in name: param = "up_proj"
        elif "down_exps" in name: param = "down_proj"
        else: continue
        expert_counts.append(shape[-1])
        total_bytes = tensor.n_bytes
        per_expert_bytes = total_bytes // shape[-1]
        ggml_type = int(tensor.tensor_type) if hasattr(tensor, "tensor_type") else 1
        shape_rest = tuple(shape[:-1])
        layer_data[layer][param] = (tensor, per_expert_bytes, shape_rest, ggml_type)
    if not layer_data:
        raise ValueError("No expert tensors found")
    n_experts = Counter(expert_counts).most_common(1)[0][0]
    all_layers = sorted(layer_data.keys())
    n_layers = len(all_layers)
    first = layer_data[all_layers[0]]
    param_specs = []
    for pname in sorted(first.keys()):
        _, per_expert_bytes, shape_rest, ggml_type = first[pname]
        param_specs.append((pname, shape_rest, ggml_type, per_expert_bytes))
    return n_layers, n_experts, param_specs, all_layers, layer_data

def write_expert_file(path, n_layers, n_experts, param_specs, ordered_layers, layer_data, fp16=False):
    # Compute per-param FP16 byte sizes
    f16_sizes = []
    for pname, shape, ggml_type, qsize in param_specs:
        if fp16 and ggml_type in (12, 14):
            f16_sizes.append(shape[0] * shape[1] * 2)  # ne0 * ne1 * 2 bytes (FP16)
        else:
            f16_sizes.append(qsize)
    
    raw_block = sum(f16_sizes)
    expert_stride = round_up(raw_block)
    file_size = HEADER_SIZE + n_layers * n_experts * expert_stride
    
    with open(path, "wb") as f:
        f.truncate(file_size)
        hdr = bytearray(HEADER_SIZE)
        hdr[:6] = MAGIC
        struct.pack_into("<I", hdr, 6, VERSION)
        off = 10
        struct.pack_into("<IIQ", hdr, off, n_layers, n_experts, expert_stride)
        off += 16
        struct.pack_into("<H", hdr, off, len(param_specs))
        off += 2
        for i, (name, shape, ggml_type, _) in enumerate(param_specs):
            dcode = DTYPE_CODE_F16 if (fp16 and ggml_type in (12,14)) else {12:DTYPE_CODE_Q4_K,14:DTYPE_CODE_Q6_K}.get(ggml_type, 2)
            name_b = name.encode("ascii")[:255]
            struct.pack_into("<B", hdr, off, len(name_b)); off += 1
            hdr[off:off+len(name_b)] = name_b; off += len(name_b)
            struct.pack_into("<B", hdr, off, len(shape)); off += 1
            for s in shape: struct.pack_into("<I", hdr, off, s); off += 4
            struct.pack_into("<B", hdr, off, dcode); off += 1
        
        # Compute per-param data offsets and strides
        param_offsets = [0] * len(param_specs)
        param_strides = [0] * len(param_specs)
        param_base = HEADER_SIZE
        for i, (_, _, _, _) in enumerate(param_specs):
            param_strides[i] = round_up(f16_sizes[i])
            param_offsets[i] = param_base
            param_base += n_layers * n_experts * param_strides[i]
        
        f.write(hdr)
        for i, (pname, shape, ggml_type, _) in enumerate(param_specs):
            stride = param_strides[i]
            base = param_offsets[i]
            for li, layer in enumerate(ordered_layers):
                tensor, pe, srest, gtype = layer_data[layer][pname]
                raw = bytes(tensor.data) if isinstance(tensor.data, memoryview) else bytes(tensor.data)
                if fp16 and ggml_type in (12, 14):
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_experts, int(pe))
                    for expert in range(n_experts):
                        f16 = dequant_to_fp16(arr[expert].tobytes(), ggml_type, shape)
                        offset = base + li * n_experts * stride + expert * stride
                        f.seek(offset)
                        f.write(f16)
                else:
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_experts, int(pe))
                    for expert in range(n_experts):
                        offset = base + li * n_experts * stride + expert * stride
                        f.seek(offset)
                        f.write(arr[expert].tobytes())
    
    print(f"\n  Done: {file_size/1024**3:.1f} GB -> {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("gguf")
    p.add_argument("out")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()
    
    print(f"Reading {args.gguf}...")
    reader = GGUFReader(args.gguf)
    n_layers, n_experts, param_specs, ordered_layers, layer_data = find_expert_params(reader)
    print(f"  Layers: {min(ordered_layers)}-{max(ordered_layers)} ({len(ordered_layers)})")
    print(f"  Experts/layer: {n_experts}")
    for name, shape, ggml_type, byte_size in param_specs:
        print(f"  Param: {name} shape={shape} type={ggml_type} {byte_size}B/expert")
    write_expert_file(args.out, n_layers, n_experts, param_specs, ordered_layers, layer_data, fp16=args.fp16)

if __name__ == "__main__":
    main()
