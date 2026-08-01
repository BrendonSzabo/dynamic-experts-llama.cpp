#!/usr/bin/env python3
"""Convert GGUF expert weights to per-expert Q4_K/Q6_K .bin file.
Handles interleaved Q4_K/Q6_K blocks (256 experts per block)."""
import sys, struct, argparse, numpy as np
from collections import defaultdict, Counter
from gguf import GGUFReader

MAGIC, VERSION, HEADER_SIZE = b"VLLM\x02", 2, 16384
QK_K = 256
DTYPE_Q4_K, DTYPE_Q6_K = 32, 34

def dequant_q4_k_block(data):
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(data[2:4], dtype=np.float16)[0].astype(np.float32)
    scales = np.frombuffer(data[4:16], dtype=np.uint8).astype(np.float32)
    qs = np.frombuffer(data[16:144], dtype=np.uint8)
    out = np.zeros(QK_K, np.float32)
    for j in range(QK_K):
        out[j] = d * scales[j//32] - dmin * (qs[j//2] >> (4*(j%2)) & 0xF)
    return out

def dequant_q6_k_block(data):
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    ql = np.frombuffer(data[2:130], dtype=np.uint8)
    qh = np.frombuffer(data[130:210], dtype=np.uint8)
    out = np.zeros(QK_K, np.float32)
    for j in range(QK_K):
        q = ((ql[j//2] >> (4*(j%2))) & 0xF) | ((qh[j//4] >> (2*(j%4))) & 0x3) << 4
        out[j] = d * (q - 32)
    return out

def dequant_full_tensor(raw, dtcode, ne0, ne1, ne):
    """Dequantize full interleaved tensor to [ne0, ne1, ne] float32."""
    bs = 144 if dtcode == DTYPE_Q4_K else 210
    deq = dequant_q4_k_block if dtcode == DTYPE_Q4_K else dequant_q6_k_block
    n_blocks = (ne0 * ne1) // QK_K
    arr = np.frombuffer(raw, dtype=np.uint8)
    total = ne0 * ne1 * ne
    out = np.zeros(total, np.float32)
    for b in range(n_blocks):
        block = arr[b*bs:(b+1)*bs]
        f32 = deq(block)
        out[b*QK_K:(b+1)*QK_K] = f32
    return out.reshape(ne0, ne1, ne)

def requant_q4_k(f32):
    """Quantize 256-element f32 to Q4_K bytes (144 bytes)."""
    out = bytearray(144)
    d = float(np.max(np.abs(f32))) / 7.5
    if d < 1e-8: d = 1.0
    dmin = float(np.min(f32))
    if dmin < -d*5: dmin = -d*5
    struct.pack_into("<e", out, 0, np.float16(d))
    struct.pack_into("<e", out, 2, np.float16(dmin))
    scales = bytearray(12)
    for si in range(12):
        sub = f32[si*32:(si+1)*32]
        sm = float(np.max(np.abs(sub - dmin))) / 15.0
        if sm < 1e-8: sm = 1.0
        scales[si] = min(255, max(1, int(sm / d * 255 + 0.5)))
    out[4:16] = bytes(scales)
    qs = bytearray(128)
    for j in range(256):
        sd = d * scales[min(j//32,11)] / 255.0
        q = int(np.clip(np.round((f32[j] - dmin) / sd), 0, 15))
        qs[j//2] |= (q & 0xF) << (4 * (j % 2))
    out[16:144] = bytes(qs)
    return bytes(out)

def requant_q6_k(f32):
    """Quantize 256 f32 to Q6_K bytes (210 bytes)."""
    out = bytearray(210)
    d = float(np.max(np.abs(f32))) / 63.0
    if d < 1e-8: d = 1.0
    struct.pack_into("<e", out, 0, np.float16(d))
    ql = bytearray(128); qh = bytearray(64)
    for j in range(256):
        q = int(np.clip(np.round(f32[j] / d + 32), 0, 63))
        ql[j//2] |= (q & 0xF) << (4 * (j % 2))
        qh[j//4] |= ((q >> 4) & 0x3) << (2 * (j % 4))
    out[2:130] = bytes(ql); out[130:210] = bytes(qh)
    return bytes(out)

def round_up(n, b=4096): return ((n + b - 1) // b) * b

def find_expert_params(reader):
    layer_data = defaultdict(dict); expert_counts = []
    for tensor in reader.tensors:
        name = tensor.name; shape = list(tensor.shape)
        if len(shape) < 2: continue
        ne = shape[-1]
        if ne < 4 or ne > 4096: continue
        layer = -1
        for part in name.split("."):
            try: layer=int(part); break
            except: pass
        if layer < 0: continue
        if "gate_up_exps" in name: param = "gate_up_proj"
        elif "gate_exps" in name: param = "gate_proj"
        elif "up_exps" in name: param = "up_proj"
        elif "down_exps" in name: param = "down_proj"
        else: continue
        expert_counts.append(ne)
        gt = int(tensor.tensor_type) if hasattr(tensor,"tensor_type") else 1
        dtcode = {12:DTYPE_Q4_K,14:DTYPE_Q6_K}.get(gt, 2)
        layer_data[layer][param] = (tensor, tuple(shape[:-1]), dtcode)
    if not layer_data: raise ValueError("No expert tensors")
    nexp = Counter(expert_counts).most_common(1)[0][0]
    layers = sorted(layer_data.keys())
    first = layer_data[layers[0]]
    specs = [(p, first[p][1], first[p][2]) for p in sorted(first.keys())]
    return len(layers), nexp, specs, layers, layer_data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("gguf"); p.add_argument("out")
    args = p.parse_args()
    r = GGUFReader(args.gguf)
    nl, ne, specs, layers, ldata = find_expert_params(r)
    print(f"Layers: {len(layers)}, Experts: {ne}")
    for s in specs: print(f"  {s[0]}: shape={s[1]} dtype={s[2]}")

    per_exp_sizes = []
    for _, shape, dtcode in specs:
        n_blocks = (shape[0] * shape[1]) // QK_K
        per_exp_sizes.append(n_blocks * (144 if dtcode==DTYPE_Q4_K else 210))

    expert_stride = round_up(sum(per_exp_sizes))
    file_size = HEADER_SIZE + nl * ne * expert_stride
    print(f"File size: {file_size/1024**3:.1f} GB")

    with open(args.out, "wb") as f:
        f.truncate(file_size)
        hdr = bytearray(HEADER_SIZE)
        hdr[:6] = MAGIC; struct.pack_into("<I", hdr, 6, VERSION); off = 10
        struct.pack_into("<IIQ", hdr, off, nl, ne, expert_stride); off += 16
        struct.pack_into("<H", hdr, off, len(specs)); off += 2
        for pname, shape, dtcode in specs:
            nb = pname.encode("ascii")[:255]
            struct.pack_into("<B", hdr, off, len(nb)); off+=1
            hdr[off:off+len(nb)] = nb; off+=len(nb)
            struct.pack_into("<B", hdr, off, len(shape)); off+=1
            for s in shape: struct.pack_into("<I", hdr, off, s); off+=4
            struct.pack_into("<B", hdr, off, dtcode); off+=1
        f.write(hdr)

        param_offsets = []; param_strides = []
        base = HEADER_SIZE
        for esz in per_exp_sizes:
            param_offsets.append(base); param_strides.append(round_up(esz))
            base += nl * ne * param_strides[-1]

        for pi, (pname, shape, dtcode) in enumerate(specs):
            stride = param_strides[pi]; pbase = param_offsets[pi]
            ne0, ne1 = shape
            print(f"  Processing {pname}...")
            for li, layer in enumerate(layers):
                tensor, _, _ = ldata[layer][pname]
                raw = bytes(tensor.data) if isinstance(tensor.data, memoryview) else bytes(tensor.data)
                f32_full = dequant_full_tensor(raw, dtcode, ne0, ne1, ne)
                for expert in range(ne):
                    exp_f32 = f32_full[:, :, expert].reshape(-1)
                    offset = pbase + li * ne * stride + expert * stride
                    f.seek(offset)
                    n_blocks = (ne0 * ne1) // QK_K
                    req = requant_q4_k if dtcode == DTYPE_Q4_K else requant_q6_k
                    for b in range(n_blocks):
                        block = exp_f32[b*QK_K:(b+1)*QK_K]
                        f.write(req(block))
    print(f"Done: {args.out}")

if __name__ == "__main__":
    main()
