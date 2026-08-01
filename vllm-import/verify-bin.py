#!/usr/bin/env python3
"""Verify .bin expert data matches GGUF source."""
import sys
import struct
import numpy as np
from gguf import GGUFReader

HEADER_SIZE = 16384
MAGIC = b'VLLM\x02\x00'  # includes null terminator from C string

def read_u32_le(data, off):
    return struct.unpack_from('<I', data, off)[0]

def read_u64_le(data, off):
    return struct.unpack_from('<Q', data, off)[0]

def read_u16_le(data, off):
    return struct.unpack_from('<H', data, off)[0]

def round_up_page(n):
    return ((n + 4095) // 4096) * 4096

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.gguf> <expert_weights.bin> [max_experts]")
        sys.exit(1)

    gguf_path = sys.argv[1]
    bin_path = sys.argv[2]
    max_check = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print(f"Reading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)

    print(f"Reading .bin: {bin_path}")
    with open(bin_path, 'rb') as f:
        bindata = f.read()

    # Parse .bin header
    assert bindata[:6] == MAGIC, "Bad magic"
    version = read_u32_le(bindata, 6)
    n_layers = read_u32_le(bindata, 10)
    n_experts = read_u32_le(bindata, 14)
    expert_stride = read_u64_le(bindata, 18)
    n_params = read_u16_le(bindata, 26)

    print(f"  version={version} layers={n_layers} experts={n_experts} params={n_params} expert_stride={expert_stride}")

    # Parse param specs
    params = []
    off = 28
    for i in range(n_params):
        name_len = bindata[off]; off += 1
        name = bindata[off:off+name_len].decode('ascii'); off += name_len
        ndim = bindata[off]; off += 1
        shape = []
        for d in range(ndim):
            shape.append(read_u32_le(bindata, off)); off += 4
        dtype_code = bindata[off]; off += 1
        params.append({'name': name, 'shape': shape, 'dtype_code': dtype_code})
        print(f"  param {i}: {name} shape={shape} dtype_code={dtype_code}")

    # Compute per-param offsets (same as llama-dyn-ex.cpp)
    GGML_TYPE_TO_DTYPE_CODE_REV = {v: k for k, v in {
        0: 2, 1: 0, 30: 1, 2: 20, 3: 21, 6: 22, 7: 23,
        8: 24, 9: 25, 10: 30, 11: 31, 12: 32, 13: 33, 14: 34, 23: 40
    }.items()}

    param_offsets = []
    param_strides = []
    data_off = HEADER_SIZE
    for p in params:
        numel = 1
        for s in p['shape']:
            numel *= s
        ggml_type = GGML_TYPE_TO_DTYPE_CODE_REV.get(p['dtype_code'], 0)
        # element size approximation
        per_exp = numel
        stride = round_up_page(per_exp)
        param_offsets.append(data_off)
        param_strides.append(stride)
        print(f"    offset={data_off} stride={stride} per_exp_bytes={per_exp}")
        data_off += n_layers * n_experts * stride

    # Find param indices
    param_idx = {p['name']: i for i, p in enumerate(params)}

    # Build GGUF expert tensor map
    expert_tensors = {}
    for tensor in reader.tensors:
        name = tensor.name
        if '_exps' not in name:
            continue
        parts = name.split('.')
        # blk.N.ffn_gate_exps.weight
        layer = int(parts[1])
        param = parts[2].replace('_exps', '')
        if param == 'gate_up':
            param = 'gate_up_proj'
        elif param == 'gate':
            param = 'gate_proj'
        elif param == 'up':
            param = 'up_proj'
        elif param == 'down':
            param = 'down_proj'
        else:
            continue
        raw = bytes(tensor.data) if hasattr(tensor.data, 'tobytes') else bytes(tensor.data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        try:
            per_exp = len(arr) // n_experts
            arr = arr.reshape(n_experts, per_exp)
        except:
            print(f"  WARNING: cannot reshape {name} shape={tensor.shape} n_experts={n_experts} bytes={len(arr)}")
            continue
        expert_tensors[(layer, param)] = arr

    print(f"\nFound {len(expert_tensors)} GGUF expert tensor entries")

    # Verify
    errors = 0
    for layer in range(min(n_layers, 2)):
        for eid in range(min(n_experts, max_check)):
            for pname in ['gate_proj', 'up_proj', 'down_proj']:
                key = (layer, pname)
                if key not in expert_tensors:
                    continue
                gguf_data = expert_tensors[key][eid].tobytes()

                pi = param_idx.get(pname)
                if pi is None:
                    continue
                bin_off = param_offsets[pi] + (layer * n_experts + eid) * param_strides[pi]
                bin_data = bindata[bin_off:bin_off + len(gguf_data)]

                match = (gguf_data == bin_data)
                if not match:
                    errors += 1
                    print(f"MISMATCH L{layer} expert{eid} {pname}: gguf={gguf_data[:8].hex()} bin={bin_data[:8].hex()}")
                else:
                    if eid < 3:
                        print(f"OK L{layer} expert{eid} {pname}: {gguf_data[:8].hex()}")

    print(f"\n{errors} mismatches found")
    if errors:
        sys.exit(1)

if __name__ == '__main__':
    main()
