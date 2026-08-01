#!/usr/bin/env python3
"""Print expert bytes from GGUF and .bin for comparison."""
import struct, sys
from gguf import GGUFReader

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwable-3.6-35b_q4_k_m.gguf"
BIN  = sys.argv[2] if len(sys.argv) > 2 else "models/qwable-weights.bin"

def dump_hex(tag, data, n=256):
    print(f"\n--- {tag} ({len(data)} bytes) ---")
    for i in range(0, min(n, len(data)), 32):
        hexl = ' '.join(f'{b:02x}' for b in data[i:i+32])
        print(f"  {i:06x}: {hexl}")

r = GGUFReader(GGUF)
for t in r.tensors:
    if t.name == "blk.0.ffn_gate_exps.weight":
        raw = bytes(t.data) if isinstance(t.data, memoryview) else bytes(t.data)
        ne = list(t.shape)[-1]
        per = len(raw) // ne
        dump_hex(f"GGUF {t.name} expert0 (of {ne}, {per}B each)", raw[0:per])
        break

with open(BIN, "rb") as f:
    magic = f.read(6)
    ver, nl, ne = struct.unpack("<III", f.read(12))
    es = struct.unpack("<Q", f.read(8))[0]
    f.seek(16384)  # skip header
    dump_hex(f"BIN {BIN} L0 E0 gate_proj", f.read(589824))

    # Verify match
    f.seek(16384)
    bin_e0 = f.read(589824)
    gguf_t = next((t for t in r.tensors if t.name == "blk.0.ffn_gate_exps.weight"), None)
    gguf_raw = bytes(gguf_t.data) if isinstance(gguf_t.data, memoryview) else bytes(gguf_t.data)
    gguf_e0 = gguf_raw[0:589824]
    print(f"\nMATCH: {gguf_e0 == bin_e0}")
    if gguf_e0 != bin_e0:
        for i in range(min(len(gguf_e0), len(bin_e0))):
            if gguf_e0[i] != bin_e0[i]:
                print(f"  First diff at byte {i}: gguf={gguf_e0[i]:02x} bin={bin_e0[i]:02x}")
                break
