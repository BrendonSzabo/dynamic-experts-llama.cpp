"""Byte-aligned expert weight file format for efficient per-expert I/O.

Layout::

    ┌──────────────────────────────────────────────────────────┐
    │  HEADER   (exactly 4096 bytes, page-aligned)             │
    │    magic:        b"VLLM\x01"              (6 bytes)      │
    │    version:      uint32_le                (4 bytes)      │
    │    n_layers:     uint32_le                (4 bytes)      │
    │    n_experts:    uint32_le                (4 bytes)      │
    │    expert_stride: uint64_le               (8 bytes)      │
    │    w13_shape:    uint32_le × 3   (hid, up, inter)        │
    │    w2_shape:     uint32_le × 3   (hid, inter, hid)       │
    │    dtype_enum:   uint8           (0=fp16, 1=bf16, 2=fp32)│
    │    reserved:     pad to 4096                              │
    ├──────────────────────────────────────────────────────────┤
    │  layer 0, expert 0  (expert_stride bytes, 4 KB aligned)  │
    │    w13_weight bytes (flat, row-major)                     │
    │    w2_weight bytes  (flat, row-major)                     │
    │    pad to 4 KB boundary                                   │
    ├──────────────────────────────────────────────────────────┤
    │  layer 0, expert 1                                       │
    │  ... layer 0, expert E-1                                  │
    │  layer 1, expert 0                                       │
    │  ... layer L-1, expert E-1                                │
    └──────────────────────────────────────────────────────────┘

Key property: every expert block has the same byte-length, so the offset
of layer *l*, expert *e* is::

    offset = 4096 + (l * n_experts + e) * expert_stride

This gives O(1) lookup — no index table needed.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import torch

_MAGIC = b"VLLM\x02"
_HEADER_SIZE = 16384
_PAGE_SIZE = mmap.ALLOCATIONGRANULARITY

_DTYPE_TABLE: dict[torch.dtype, int] = {
    torch.float16: 0, torch.bfloat16: 1, torch.float32: 2,
    torch.float8_e4m3fn: 3, torch.float8_e5m2: 4,
    torch.int8: 10, torch.int32: 11,
}
_DTYPE_REVERSE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TABLE.items()}


def _round_up(n: int, boundary: int = _PAGE_SIZE) -> int:
    return ((n + boundary - 1) // boundary) * boundary


def _dtype_bytes(dtype: torch.dtype) -> int:
    try:
        return torch.finfo(dtype).bits // 8
    except TypeError:
        return dtype.itemsize


@dataclass
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int = 0
    byte_offset: int = 0

    def __post_init__(self):
        self.numel = 1
        for s in self.shape:
            self.numel *= s


# ── writer ────────────────────────────────────────────────────────


def write_dynamic_expert_file(
    path: str | Path,
    expert_weights: Generator[tuple[int, int, tuple[torch.Tensor, ...]], None, None],
    n_layers: int,
    n_experts: int,
    params: list[ParamSpec],
) -> int:
    raw_block = sum(p.numel * _dtype_bytes(p.dtype) for p in params)
    expert_stride = _round_up(raw_block)
    file_size = _HEADER_SIZE + n_layers * n_experts * expert_stride

    with open(path, "wb") as f:
        f.truncate(file_size)
        hdr = bytearray(_HEADER_SIZE)
        struct.pack_into("<6sI", hdr, 0, _MAGIC, 2)
        off = 10
        struct.pack_into("<IIQ", hdr, off, n_layers, n_experts, expert_stride)
        off += 16
        struct.pack_into("<H", hdr, off, len(params))
        off += 2
        for p in params:
            name_b = p.name.encode("ascii")[:255]
            struct.pack_into("<B", hdr, off, len(name_b))
            off += 1
            hdr[off : off + len(name_b)] = name_b
            off += len(name_b)
            struct.pack_into("<B", hdr, off, len(p.shape))
            off += 1
            struct.pack_into("<" + "I" * len(p.shape), hdr, off, *p.shape)
            off += 4 * len(p.shape)
            struct.pack_into("<B", hdr, off, _DTYPE_TABLE.get(p.dtype, 255))
            off += 1
        f.write(hdr)

        pad = b"\x00" * (expert_stride - raw_block)
        for layer, expert, tensors in expert_weights:
            data_offset = _HEADER_SIZE + (layer * n_experts + expert) * expert_stride
            f.seek(data_offset)
            for t in tensors:
                f.write(t.contiguous().cpu().numpy().tobytes())
            f.write(pad)

    return file_size


# ── reader ─────────────────────────────────────────────────────────


class DynamicExpertFileReader:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._fd = os.open(str(self._path), os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

        magic = self._mm[:6]
        if magic != _MAGIC:
            raise ValueError(f"Bad magic: {magic!r}, expected {_MAGIC!r}")
        _ver = struct.unpack_from("<I", self._mm, 6)[0]
        if _ver != 2:
            raise ValueError(f"Unsupported version: {_ver}")
        self.n_layers, self.n_experts, self.expert_stride = struct.unpack_from(
            "<IIQ", self._mm, 10
        )
        off = 26
        n_params = struct.unpack_from("<H", self._mm, off)[0]
        off += 2
        self.params = []
        for _ in range(n_params):
            name_len = self._mm[off]
            off += 1
            name = self._mm[off : off + name_len].decode("ascii")
            off += name_len
            ndim = self._mm[off]
            off += 1
            shape = struct.unpack_from("<" + "I" * ndim, self._mm, off)
            off += 4 * ndim
            dtype = _DTYPE_REVERSE.get(self._mm[off], torch.float16)
            off += 1
            self.params.append(ParamSpec(name, shape, dtype))
        self._compute_offsets()

    def _compute_offsets(self):
        byte_off = 0
        for p in self.params:
            p.byte_offset = byte_off
            byte_off += p.numel * _dtype_bytes(p.dtype)

    def _expert_offset(self, layer: int, expert: int) -> int:
        return _HEADER_SIZE + (layer * self.n_experts + expert) * self.expert_stride

    def read_expert(
        self, layer: int, expert: int
    ) -> dict[str, torch.Tensor]:
        off = self._expert_offset(layer, expert)
        result = {}
        for p in self.params:
            result[p.name] = torch.frombuffer(
                self._mm, dtype=p.dtype, count=p.numel,
                offset=off + p.byte_offset,
            ).reshape(p.shape)
        return result

    def read_expert_copy(
        self, layer: int, expert: int,
    ) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.read_expert(layer, expert).items()}

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
