# .cap Binary Format Specification

Version 1.  All integers little-endian.

## File Header (28 bytes)

| Offset | Size | Type   | Field          | Description                            |
|--------|------|--------|----------------|----------------------------------------|
| 0      | 4    | u32    | magic          | `0x44434552` ("RECD")                  |
| 4      | 4    | u32    | version        | `1`                                    |
| 8      | 4    | u32    | n_embd         | model embedding dimension              |
| 12     | 4    | u32    | n_vocab        | vocabulary size                        |
| 16     | 4    | u32    | n_layer        | total number of layers                 |
| 20     | 4    | u32    | n_experts      | total expert count (MoE models)        |
| 24     | 4    | u32    | n_expert_used  | experts routed per token               |

## Per-Token Record

One record per decoded token.  Written sequentially after the header.
The file is a flat sequence: `header | record_0 | record_1 | ... | record_N-1`.

### Record Header (12 bytes)

| Offset | Size | Type | Field       | Description                                           |
|--------|------|------|-------------|-------------------------------------------------------|
| 0      | 4    | i32  | token_id    | raw token id (decoded output token)                   |
| 4      | 4    | i32  | position    | cumulative token position in the sequence             |
| 8      | 4    | u32  | n_tensors   | number of tensor entries following                   |

### Tensor Entry (repeated n_tensors times)

| Offset | Size     | Type    | Field      | Description                                 |
|--------|----------|---------|------------|---------------------------------------------|
| 0      | 4        | u32     | name_len   | length of name string in bytes              |
| 4      | name_len | char[]  | name       | tensor name (not null-terminated)           |
|        | 4        | i32     | il         | layer index (-1 = global, 0..n_layer-1)     |
|        | 4        | u32     | n_dims     | number of dimensions (1-4)                  |
|        | n_dims*8 | i64[]   | ne         | shape array (ne[0]..ne[n_dims-1])           |
|        | 4        | u32     | dtype      | ggml_type enum value (see Dtype Table)      |
|        | data_bytes | raw   | data       | tensor payload, original dtype              |

The tensor name is a raw byte string, `name_len` bytes long, NOT null-terminated.

`il = -1` means the tensor is global (e.g. embeddings, final output).
`il >= 0` is the 0-based layer index.

`n_dims` is always between 1 and 4.  `ne[0]` through `ne[n_dims-1]` are valid;
remaining `ne[]` entries are undefined and should be ignored.

`data_bytes` is computed from the shape and dtype:

```
nelements = prod(ne[0..n_dims-1])
elem_size = ggml_type_size(dtype)
data_bytes = nelements * elem_size
```

Note: the captured tensors are activations (F32 or F16); quantized weight tensors
are excluded.  See Dtype Table for the element sizes per type.

**Important**: the `ne[]` values in the file are the LIVE tensor shapes at decode
time (single-token batch), not the warmup worst-case shapes.  The data payload
matches these live shapes.

## Dtype Table (ggml_type enum)

| Value | Name       | Element Size (bytes) | Notes                       |
|-------|------------|----------------------|-----------------------------|
| 0     | F32        | 4                    | float32                     |
| 1     | F16        | 2                    | float16 (IEEE 754 half)     |
| 2     | Q4_0       | 32                   | quantized, not in captures  |
| 3     | Q4_1       | 32                   | quantized, not in captures  |
| 6     | Q5_0       | 32                   | quantized, not in captures  |
| 7     | Q5_1       | 32                   | quantized, not in captures  |
| 8     | Q8_0       | 34                   | quantized, not in captures  |
| 9     | Q8_1       | 34                   | quantized, not in captures  |
| 10    | Q2_K       | 256                  | quantized, not in captures  |
| 11    | Q3_K       | 256                  | quantized, not in captures  |
| 12    | Q4_K       | 256                  | quantized, not in captures  |
| 13    | Q5_K       | 256                  | quantized, not in captures  |
| 14    | Q6_K       | 256                  | quantized, not in captures  |
| 15    | Q8_K       | 256                  | quantized, not in captures  |

In practice, activations are always F32 (dtype=0, 4 bytes/element) or F16
(dtype=1, 2 bytes/element).  Quantized types only appear in weight tensors,
which are excluded from capture.

## Tensor Names

The tensors in each record correspond to the named intermediate tensors produced
during the model's forward pass.  They are registered by the `cb()` graph-building
callback, which fires for every computed (non-input, non-view) tensor.

Common tensor names across MoE transformer layers:

| Name              | il    | Typical Shape (1 token) | Description                     |
|-------------------|-------|-------------------------|---------------------------------|
| `embd`            | -1    | [n_embd]                | token embedding output          |
| `inp_embd`        | -1    | [n_embd]                | embedding input (after scaling) |
| `attn_norm`       | 0..39 | [n_embd]                | RMS norm before attention       |
| `ffn_inp`         | 0..39 | [n_embd]                | attention output + residual     |
| `ffn_norm`        | 0..39 | [n_embd]                | RMS norm before FFN             |
| `ffn_moe_gate`    | 0..39 | [n_ff, n_expert_used]   | MoE gate activation per expert  |
| `ffn_moe_up`      | 0..39 | [n_ff, n_expert_used]   | MoE up-projection per expert    |
| `ffn_moe_down`    | 0..39 | [n_embd, n_expert_used] | MoE down-projection per expert  |
| `ffn_moe_out`     | 0..39 | [n_embd]                | aggregated MoE output           |
| `ffn_out`         | 0..39 | [n_embd]                | after shared expert + residual  |
| `l_out`           | 0..39 | [n_embd]                | layer output (next layer input) |
| `result_norm`     | -1    | [n_embd]                | final RMS norm output           |
| `result_output`   | -1    | [n_vocab]               | logits                          |

Additional model-specific tensors (attention intermediates: q, k, v, kq, attn
output; SSM/conv states for recurrent layers; fused-op intermediates) are also
captured depending on the model architecture.  The exact set of tensors in each
record is determined by the graph topology during warmup.

## Parsing Pseudocode

```python
import struct

def read_cap(path):
    with open(path, 'rb') as f:
        # header
        magic, ver, n_embd, n_vocab, n_layer, n_experts, n_exp_used = \
            struct.unpack('<7I', f.read(28))
        assert magic == 0x44434552 and ver == 1

        dtype_sizes = {0: 4, 1: 2}  # F32=4, F16=2

        records = []
        while True:
            header = f.read(12)
            if len(header) < 12: break
            token_id, pos, n_tensors = struct.unpack('<iiI', header)

            tensors = {}
            for _ in range(n_tensors):
                name_len = struct.unpack('<I', f.read(4))[0]
                name = f.read(name_len).decode()
                il, n_dims = struct.unpack('<iI', f.read(8))
                ne = struct.unpack(f'<{n_dims}q', f.read(n_dims * 8))
                dtype = struct.unpack('<I', f.read(4))[0]

                nelements = 1
                for d in range(n_dims):
                    nelements *= ne[d]
                data_bytes = nelements * dtype_sizes.get(dtype, 1)
                data = f.read(data_bytes)

                tensors[name] = {
                    'il': il, 'shape': ne[:n_dims], 'dtype': dtype,
                    'data': data, 'nbytes': data_bytes
                }

            records.append({
                'token_id': token_id, 'position': pos,
                'n_tensors': n_tensors, 'tensors': tensors
            })

    return records
```

## Current Limitations

- **Entry count is inflated**: the warmup graph build registers tensors for
  worst-case batch sizes.  The recorded shapes (in `ne[]`) are correct for the
  actual decode batch, but the number of tensor entries per record includes
  stale entries from early warmup passes.  Approximately 5000 entries per token
  for a 40-layer model.
- **No slot/sequence metadata**: records from parallel sequence generation are
  interleaved without identifying which sequence they belong to.
- **No timestamp or generation metadata**: the file contains no wall-clock time,
  total token count, or other session-level info beyond the header.
