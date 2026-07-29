#!/usr/bin/env python3
"""
Convert GGUF model expert weights to the llm_arch_expert binary format.

Usage:
    python convert-gguf-to-expert-binary.py model.gguf expert_weights.bin

Auto-discovers expert weight params by finding tensors with shape[0] == n_experts.
Writes to the same binary format as vllm-import/dynamic_experts/file_format.py.

Requirements:
    pip install gguf numpy
"""

import sys
import struct
from collections import defaultdict, Counter
import numpy as np
from gguf import GGUFReader

MAGIC = b"VLLM\x02"
VERSION = 2
HEADER_SIZE = 16384
PAGE_SIZE = 4096

# File-format dtype codes (must match llama-moe-expert-file.cpp)
GGML_TYPE_TO_DTYPE_CODE = {
    0:  2,   # F32
    1:  0,   # F16
    30: 1,   # BF16
    2:  20,  # Q4_0
    3:  21,  # Q4_1
    6:  22,  # Q5_0
    7:  23,  # Q5_1
    8:  24,  # Q8_0
    9:  25,  # Q8_1
    10: 30,  # Q2_K
    11: 31,  # Q3_K
    12: 32,  # Q4_K
    13: 33,  # Q5_K
    14: 34,  # Q6_K
    23: 40,  # IQ4_NL
}


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
        # Expert count is the last dimension for Qwen-style GGUF layout
        # [hidden_dim, inter_dim, n_experts] or [inter_dim, hidden_dim, n_experts]
        n_exp_dim = shape[-1]
        if n_exp_dim < 4 or n_exp_dim > 4096:
            continue

        # Parse layer number
        layer = -1
        for part in name.split("."):
            try:
                layer = int(part)
                break
            except ValueError:
                pass
        if layer < 0:
            continue

        # Param name — only match actual expert weight tensors
        if "gate_up_exps" in name:
            param = "gate_up_proj"
        elif "gate_exps" in name:
            param = "gate_proj"
        elif "up_exps" in name:
            param = "up_proj"
        elif "down_exps" in name:
            param = "down_proj"
        else:
            continue  # skip non-expert tensors (gate_inp, etc.)

        expert_counts.append(shape[-1])
        total_bytes = tensor.n_bytes
        per_expert_bytes = total_bytes // shape[-1]
        ggml_type = int(tensor.tensor_type) if hasattr(tensor, "tensor_type") else 1
        dtype_code = GGML_TYPE_TO_DTYPE_CODE.get(ggml_type, 2)
        shape_rest = tuple(shape[:-1])

        layer_data[layer][param] = (tensor, per_expert_bytes, shape_rest, dtype_code)

    if not layer_data:
        raise ValueError("No expert tensors found")

    n_experts = Counter(expert_counts).most_common(1)[0][0]
    all_layers = sorted(layer_data.keys())
    n_layers = len(all_layers)

    first = layer_data[all_layers[0]]
    param_specs = []
    for pname in sorted(first.keys()):
        _, per_expert_bytes, shape_rest, dtype_code = first[pname]
        param_specs.append((pname, shape_rest, dtype_code, per_expert_bytes))

    return n_layers, n_experts, param_specs, all_layers, layer_data


def write_expert_file(path, n_layers, n_experts, param_specs,
                      ordered_layers, layer_data):
    raw_block = sum(ps[3] for ps in param_specs)
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
        for name, shape, dtype_code, _ in param_specs:
            name_b = name.encode("ascii")[:255]
            struct.pack_into("<B", hdr, off, len(name_b))
            off += 1
            hdr[off : off + len(name_b)] = name_b
            off += len(name_b)
            struct.pack_into("<B", hdr, off, len(shape))
            off += 1
            for s in shape:
                struct.pack_into("<I", hdr, off, s)
                off += 4
            struct.pack_into("<B", hdr, off, dtype_code)
            off += 1
        f.write(hdr)

        # Per-param layout: each param's data for ALL experts is contiguous
        # [up_proj for all experts * n_layers][gate_proj for all * n_layers][down_proj for all * n_layers]
        # Within each param section: [layer0_expert0, layer0_expert1, ..., layer0_expertN-1, layer1_expert0, ...]
        param_base_off = HEADER_SIZE
        for pname, shape, _, _ in param_specs:
            per_exp = layer_data[ordered_layers[0]][pname][1]
            param_stride = round_up(per_exp)
            for li, layer in enumerate(ordered_layers):
                tensor, pe, _, _ = layer_data[layer][pname]
                raw = bytes(tensor.data) if isinstance(tensor.data, memoryview) else bytes(tensor.data)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_experts, int(pe))
                for expert in range(n_experts):
                    offset = param_base_off + li * n_experts * param_stride + expert * param_stride
                    f.seek(offset)
                    f.write(arr[expert].tobytes())
            param_base_off += n_layers * n_experts * param_stride

    print(f"\n  Done: {file_size/1024**3:.1f} GB -> {path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    gguf, out = sys.argv[1], sys.argv[2]

    print(f"Reading {gguf}...")
    reader = GGUFReader(gguf)

    n_layers, n_experts, param_specs, ordered_layers, layer_data = \
        find_expert_params(reader)

    print(f"  Layers: {min(ordered_layers)}-{max(ordered_layers)} "
          f"({len(ordered_layers)} layers)")
    print(f"  Experts/layer: {n_experts}")
    for name, shape, dtype_code, byte_size in param_specs:
        print(f"  Param: {name} shape={shape} dtype={dtype_code} "
              f"{byte_size}B/expert")

    write_expert_file(out, n_layers, n_experts,
                      param_specs, ordered_layers, layer_data)


if __name__ == "__main__":
    main()
