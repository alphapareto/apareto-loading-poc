"""Kernel-ready, layer-contiguous expert store: a layer streams to the GPU in one read.

Built once from the checkpoint-layout PackedStore. Each record holds one expert exactly as the
FlashInfer CUTLASS pool consumes it (w13 as [up, gate], block scales swizzled), so a load is a
read plus DMA, with no CPU repack. 40 contiguous layer blocks of 256 records.

Record order inside a block:
- version 2, `expert_id` (current): expert e of layer l is record e. The file depends only on the
  checkpoint, not on a ranking; NVMe reads a scattered subset about as fast
  as a contiguous one, so no ordering is needed for speed.
- version 1, `global_rank` (superseded, not served): records in a global seat order.
"""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from .store import Ranking, align_up, decode_json, digest, expert_format, header_digest, read_at

MAGIC = b"APKL"
VERSION = 2
ORDERS = ("expert_id", "global_rank")
TRANSFORM = "flashinfer_cutlass_nvfp4_v1: w13 [up,gate]; block scales swizzle 128x4; global13 = gate global"
# Kernel-ready fields in record order: name, dtype, element bytes.
FIELDS = (("w13", "uint8", 1), ("w2", "uint8", 1), ("s13", "float8_e4m3fn", 1),
          ("s2", "float8_e4m3fn", 1), ("global13", "float32", 4), ("global2", "float32", 4))
FIELD_ALIGN = 256


def field_shapes(schema):
    packed = expert_format(schema)[1]["packed"]
    middle, packed_hidden = schema[f"gate_proj.{packed}"]["shape"]
    hidden = packed_hidden * 2
    pad = lambda value, to: (value + to - 1) // to * to
    return {"w13": [2 * middle, hidden // 2], "w2": [hidden, middle // 2],
            "s13": [pad(2 * middle, 128), pad(hidden // 16, 4)], "s2": [pad(hidden, 128), pad(middle // 16, 4)],
            "global13": [], "global2": []}


def record_layout(schema):
    shapes, layout, offset = field_shapes(schema), {}, 0
    for name, dtype, size in FIELDS:
        nbytes = int(np.prod(shapes[name], dtype=np.int64)) * size if shapes[name] else size
        layout[name] = {"offset": offset, "nbytes": nbytes, "dtype": dtype, "shape": shapes[name]}
        offset = align_up(offset + nbytes, FIELD_ALIGN)
    return layout, offset


def layer_orders(ranking, order="expert_id"):
    """Per layer, the expert id stored at each record position."""
    if order == "expert_id":
        return [list(range(ranking.experts)) for _ in range(ranking.layers)]
    if order != "global_rank":
        raise ValueError(f"Unknown layer-store record order: {order}")
    orders = [[] for _ in range(ranking.layers)]
    for seat in ranking.order:
        layer, expert = divmod(int(seat), ranking.experts)
        orders[layer].append(expert)
    return orders


def build_layer_store(packed, output, alignment=4096, progress=None, order="expert_id"):
    """Write the kernel-ready store from a verified PackedStore; refuses an existing output path."""
    import torch
    from .vllm027.prefill_weights import HostWeights
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    ranking = packed.ranking
    layout, record_bytes = record_layout(packed.schema)
    stride = align_up(record_bytes, alignment)
    orders = layer_orders(ranking, order)
    header = {"magic": MAGIC.decode(), "version": VERSION, "record_order": order, "transform": TRANSFORM,
              "layers": ranking.layers, "experts_per_layer": ranking.experts,
              "source_store_header_sha256": packed.header["header_sha256"], "align": alignment,
              "record_layout": layout, "record_bytes": record_bytes, "record_stride": stride,
              "layer_bytes": stride * ranking.experts, "layer_order": orders,
              "record_sha256": None, "data_offset": None}
    # Fix the header size before checksums are known: 64 hex chars per record.
    header["record_sha256"] = ["0" * 64] * (ranking.layers * ranking.experts)
    header["data_offset"] = 0
    probe = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    data_offset = align_up(8 + len(probe) + 4096, alignment)
    header["data_offset"] = data_offset
    checksums = []
    with partial.open("xb") as handle:
        handle.truncate(data_offset + ranking.layers * header["layer_bytes"])
        for layer in range(ranking.layers):
            records = [packed.tensors(packed.read_ranks(packed.rank_of(layer, expert), 1)) for expert in orders[layer]]
            host = HostWeights.from_records(records, packed.schema)
            block = np.zeros((ranking.experts, stride), dtype=np.uint8)
            for name, field in layout.items():
                tensor = getattr(host, name).contiguous()
                raw = tensor.view(torch.uint8).reshape(ranking.experts, -1).numpy()
                if raw.shape[1] != field["nbytes"]:
                    raise ValueError(f"Kernel-ready {name} has {raw.shape[1]} bytes, expected {field['nbytes']}")
                block[:, field["offset"]:field["offset"] + field["nbytes"]] = raw
            checksums.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in block)
            handle.seek(data_offset + layer * header["layer_bytes"])
            handle.write(block.tobytes())
            if progress:
                progress(layer)
        header["record_sha256"] = checksums
        header["header_sha256"] = header_digest(header)
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if 8 + len(raw) > data_offset:
            raise ValueError("Layer-store header outgrew its reserved space")
        handle.seek(0)
        handle.write(MAGIC + struct.pack("<I", len(raw)) + raw)
        handle.flush()
        os.fsync(handle.fileno())
    partial.rename(output)
    return header


class LayerStore:
    def __init__(self, path, ranking=None, source_header_sha256=None):
        self.path = Path(path)
        self.ranking = ranking if ranking is not None else Ranking.expert_order()
        self.fd = os.open(self.path, os.O_RDONLY)
        try:
            prefix = read_at(self.fd, 8, 0)
            if prefix[:4] != MAGIC:
                raise ValueError("Missing layer-store magic")
            length = struct.unpack("<I", prefix[4:])[0]
            h = self.header = decode_json(read_at(self.fd, length, 8))
            if h["version"] not in (1, VERSION) or h["transform"] != TRANSFORM or h["header_sha256"] != header_digest(h):
                raise ValueError("Unsupported or corrupt layer-store header")
            self.record_order = h.get("record_order", "global_rank")
            if (h["layers"], h["experts_per_layer"]) != (self.ranking.layers, self.ranking.experts):
                raise ValueError("Layer-store geometry mismatch")
            if source_header_sha256 is not None and h["source_store_header_sha256"] != source_header_sha256:
                raise ValueError("Layer store was built from a different packed store")
            if h["layer_order"] != layer_orders(self.ranking, self.record_order):
                raise ValueError(f"Layer-store records are not in {self.record_order} order")
            self.layout, self.stride, self.layer_bytes = h["record_layout"], h["record_stride"], h["layer_bytes"]
            self.data_offset = h["data_offset"]
            if os.fstat(self.fd).st_size != self.data_offset + self.ranking.layers * self.layer_bytes:
                raise ValueError("Layer-store size disagrees with its header")
            self.orders = [np.asarray(order, dtype=np.int64) for order in h["layer_order"]]
            self.positions = np.empty((self.ranking.layers, self.ranking.experts), dtype=np.int64)
            for layer, order in enumerate(self.orders):
                self.positions[layer, order] = np.arange(self.ranking.experts)
            self._verified = np.zeros(self.ranking.layers * self.ranking.experts, dtype=bool)
            self._identity = self._file_identity()
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        os.close(self.fd)

    def _file_identity(self):
        stat = os.fstat(self.fd)
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def read_into(self, layer, first, count, buffer, drop_cache=False, verify=True):
        """One read of records [first, first+count) of a layer into `buffer` (a writable uint8 array).

        drop_cache: evict the bytes just read from the page cache. The host tier copies the whole store into
        pinned memory once; keeping a second 18 GB in the cache only adds reclaim pressure.
        verify=False: skip the per-record checksum (used for the host tier's NVMe misses, where it halves the miss
        cost; the file identity is still checked on every read).
        """
        if not 0 <= layer < self.ranking.layers or first < 0 or count < 1 or first + count > self.ranking.experts:
            raise ValueError("Layer-store read outside its layer")
        if self._file_identity() != self._identity:
            raise ValueError("Layer-store file changed after it was opened")
        size = count * self.stride
        view = memoryview(buffer)[:size]
        offset = self.data_offset + layer * self.layer_bytes + first * self.stride
        done = 0
        while done < size:
            got = os.preadv(self.fd, [view[done:]], offset + done)
            if got <= 0:
                raise ValueError("Short layer-store read")
            done += got
        if drop_cache:
            os.posix_fadvise(self.fd, offset, size, os.POSIX_FADV_DONTNEED)
        if not verify:
            return size
        # Each record is checksummed before its first use in this process.
        base = layer * self.ranking.experts + first
        for index in np.flatnonzero(~self._verified[base:base + count]):
            record = view[index * self.stride:(index + 1) * self.stride]
            if digest(record) != self.header["record_sha256"][base + index]:
                raise ValueError(f"Layer-store checksum mismatch at layer {layer} record {first + index}")
            self._verified[base + index] = True
        return size


def inverse_record(record, layout, schema):
    """Checkpoint-layout fields from one kernel-ready record (round-trip check)."""
    fields = expert_format(schema)[1]
    packed, block, glob = fields["packed"], fields["block_scale"], fields["global_scale"]
    middle, packed_hidden = schema[f"gate_proj.{packed}"]["shape"]
    hidden = packed_hidden * 2
    def field(name):
        f = layout[name]
        return np.frombuffer(record, dtype=np.uint8, count=f["nbytes"], offset=f["offset"])
    def unswizzle(raw, rows, columns, padded_rows, padded_columns):
        tiles = raw.reshape(padded_rows // 128, padded_columns // 4, 32, 4, 4)
        return tiles.transpose(0, 3, 2, 1, 4).reshape(padded_rows, padded_columns)[:rows, :columns]
    w13 = field("w13").reshape(2 * middle, hidden // 2)
    s13 = unswizzle(field("s13"), 2 * middle, hidden // 16, *layout["s13"]["shape"])
    s2 = unswizzle(field("s2"), hidden, middle // 16, *layout["s2"]["shape"])
    return {f"gate_proj.{packed}": w13[middle:].tobytes(), f"up_proj.{packed}": w13[:middle].tobytes(),
            f"down_proj.{packed}": field("w2").tobytes(),
            f"gate_proj.{block}": s13[middle:].tobytes(), f"up_proj.{block}": s13[:middle].tobytes(),
            f"down_proj.{block}": s2.tobytes(),
            f"gate_proj.{glob}": field("global13").tobytes(),
            f"down_proj.{glob}": field("global2").tobytes()}


def check_source_header(path, ranking, checkpoint):
    """The header of the store the layer store was built from (saved as JSON once that store was deleted):
    verifies its own digest and geometry, then binds the served checkpoint to it by tensor schema and
    weight-byte fingerprint. Returns the header; its header_sha256 is what the layer store records."""
    h = decode_json(Path(path).read_bytes())
    if h.get("header_sha256") != header_digest(h):
        raise ValueError("Source header checksum mismatch")
    if (h["layers"], h["experts_per_layer"]) != (ranking.layers, ranking.experts):
        raise ValueError("Source header geometry mismatch")
    if checkpoint.schema != h["tensor_schema"] or checkpoint.fingerprint() != h["source_fingerprint"]:
        raise ValueError("Checkpoint does not match the expert store's source weight bytes/schema")
    return h
