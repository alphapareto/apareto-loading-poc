"""Validated, byte-preserving safetensors packing and rank-ordered store access.

No model or CUDA allocation is needed. Quantization labels and config hashes are
provenance, not compatibility gates; tensor geometry and actual bytes are checked.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def artifacts_dir():
    """The R* artifacts directory: APARETO_ARTIFACTS, else the repository's artifacts/ (as scripts/rstar_emit.py)."""
    return Path(os.environ.get("APARETO_ARTIFACTS", ROOT / "artifacts"))
MAGIC = b"APEX"
VERSION = 2
MAX_HEADER_BYTES = 64 << 20
EXPERT_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
TEXT_EXPERT_RE = re.compile(r"^model\.(?:language_model\.)?layers\.\d+\.mlp\.experts\.")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# The two NVFP4 builds of this model carry byte-identical expert payloads under different suffixes
# (verified: seat payload 1,769,496 bytes and stride 1,773,568 on both). Detection is on the suffix
# set the checkpoint actually has, never on config.json's quant_method, so a store header alone is
# enough to name its own fields.
EXPERT_FORMATS = {
    "compressed-tensors": {"packed": "weight_packed", "block_scale": "weight_scale",
                           "global_scale": "weight_global_scale", "input_scale": "input_global_scale"},
    "modelopt": {"packed": "weight", "block_scale": "weight_scale",
                 "global_scale": "weight_scale_2", "input_scale": "input_scale"},
}


# How the per-tensor global scale enters dequantization. Measured on seats (0,0) and
# (20,100): modelopt's weight_scale_2 MULTIPLIES where compressed-tensors' weight_global_scale
# DIVIDES, and the two are near-reciprocals (31104 vs 3.215e-05). Getting this backwards does not
# crash -- it produces weights 9 orders of magnitude too large that still correlate 0.9998 with the
# truth, so it must never be inferred.
EXPERT_SCALE_CONVENTION = {"compressed-tensors": "divide", "modelopt": "multiply"}


def expert_format(schema):
    """Name the per-expert tensor suffixes this checkpoint or store header uses."""
    suffixes = {name.split(".", 1)[1] for name in schema if "." in name}
    matched = [name for name, fields in EXPERT_FORMATS.items() if set(fields.values()) <= suffixes]
    if len(matched) != 1:
        raise ValueError(f"Expert tensor layout matched {matched or 'nothing'}: {sorted(suffixes)}")
    return matched[0], dict(EXPERT_FORMATS[matched[0]])


def expert_field(schema, projection, role):
    return f"{projection}.{expert_format(schema)[1][role]}"


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4,
    "U64": 8, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def align_up(value, alignment):
    integer(alignment, "alignment", 1)
    if alignment & (alignment - 1):
        raise ValueError("Alignment must be a power of two")
    return (value + alignment - 1) // alignment * alignment


def tensor_size(dtype, shape):
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported safetensors storage dtype: {dtype}")
    if not isinstance(shape, list):
        raise ValueError("Tensor shape must be a list")
    for dimension in shape:
        integer(dimension, "tensor dimension")
    return math.prod(shape) * DTYPE_BYTES[dtype]


def read_at(fd, size, offset):
    """One contiguous file range, filled directly without a second full-size copy."""
    result = bytearray(size)
    view = memoryview(result)
    done = 0
    while done < size:
        amount = os.preadv(fd, [view[done:]], offset + done)
        if amount == 0:
            raise ValueError(f"Truncated file at byte {offset + done}; needed {size - done} more bytes")
        done += amount
    return result


def header_digest(header):
    return digest(canonical({**header, "header_sha256": "0" * 64}))


@dataclass(frozen=True)
class Ranking:
    layers: int
    experts: int
    order: np.ndarray

    def __post_init__(self):
        integer(self.layers, "layers", 1)
        integer(self.experts, "experts", 1)
        array = np.asarray(self.order)
        count = self.layers * self.experts
        if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Ranking must be one integer entry per seat")
        if not np.array_equal(np.sort(array), np.arange(count)):
            raise ValueError("Ranking must be a total-order permutation of every seat")
        array = array.astype("<i4", copy=True)
        array.setflags(write=False)
        object.__setattr__(self, "order", array)

    @property
    def sha256(self):
        return digest(self.order.tobytes())

    @classmethod
    def expert_order(cls, layers=40, experts=256):
        """Every seat in expert-id order (layer 0 expert 0, layer 0 expert 1, ...), the order the expert stores are
        built in. 40 layers x 256 experts is the one requirement on the checkpoint; Checkpoint checks it."""
        return cls(layers, experts, np.arange(layers * experts))


@dataclass(frozen=True)
class TensorRange:
    shard: str
    name: str
    dtype: str
    shape: tuple
    offset: int
    nbytes: int


class Checkpoint:
    """Checked tensor ranges with bounded byte reads; supports the verified per-expert layouts."""
    def __init__(self, model, ranking):
        self.model = Path(model).resolve()
        self.store_root = Path(os.environ.get("APARETO_MODEL_STORE", self.model.parent)).resolve()
        self.ranking = ranking
        self.stack = ExitStack()
        self.files = {}
        self.seats = {}
        self.shards = {}
        self._fingerprint = None
        # Set by a caller to reuse a full fingerprint while the shard files keep their identity.
        self.fingerprint_cache_dir = None
        self.fingerprint_source = None
        try:
            config_bytes = (self.model / "config.json").read_bytes()
            config = decode_json(config_bytes)
            text = config.get("text_config", config)
            if (text["num_hidden_layers"], text["num_experts"]) != (ranking.layers, ranking.experts):
                raise ValueError("Checkpoint geometry does not match the ranking")
            index_path = self.model / "model.safetensors.index.json"
            index = decode_json(index_path.read_bytes()) if index_path.exists() else None
            names = sorted(set(index["weight_map"].values())) if index else sorted(
                path.name for path in self.model.glob("*.safetensors"))
            if not names:
                raise ValueError("No safetensors shards found")
            weight_map = {}
            manifest = {}
            unsupported = []
            for shard in names:
                if not isinstance(shard, str):
                    raise ValueError("Shard name must be text")
                path = (self.model / shard).resolve()
                # A shard may be a symlink into a sibling checkpoint directory (a variant that shares its
                # weight files with another). Containment is kept, one level wider: the target must
                # still resolve inside the model store, so ../../etc/passwd is refused.
                if not (path.is_relative_to(self.model) or path.is_relative_to(self.store_root)):
                    raise ValueError(
                        f"Shard escapes the model store: {shard} -> {path} "
                        f"(allowed: {self.model} or {self.store_root})")
                source = self.stack.enter_context(path.open("rb"))
                self.files[shard] = source
                stat = os.fstat(source.fileno())
                prefix = source.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Missing safetensors header: {shard}")
                length = struct.unpack("<Q", prefix)[0]
                if not 0 < length <= min(MAX_HEADER_BYTES, stat.st_size - 8):
                    raise ValueError(f"Invalid safetensors header length: {shard}")
                header = decode_json(source.read(length))
                base = 8 + length
                ranges = []
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    if name in weight_map:
                        raise ValueError(f"Duplicate tensor across shards: {name}")
                    begin, end = entry["data_offsets"]
                    integer(begin, "tensor start")
                    integer(end, "tensor end")
                    nbytes = tensor_size(entry["dtype"], entry["shape"])
                    if end - begin != nbytes or end > stat.st_size - base:
                        raise ValueError(f"Tensor shape/dtype/range disagrees: {name}")
                    ranges.append((begin, end))
                    weight_map[name] = shard
                    manifest[name] = [shard, entry["dtype"], entry["shape"], begin, end]
                    match = EXPERT_RE.fullmatch(name)
                    if match:
                        layer, expert, suffix = match.groups()
                        seat = self.seats.setdefault((int(layer), int(expert)), {})
                        if suffix in seat:
                            raise ValueError(f"Duplicate expert tensor: {name}")
                        seat[suffix] = TensorRange(shard, name, entry["dtype"], tuple(entry["shape"]),
                                                  base + begin, nbytes)
                    elif TEXT_EXPERT_RE.match(name):
                        unsupported.append(name)
                position = 0
                for begin, end in sorted(ranges):
                    if begin != position:
                        raise ValueError(f"Noncontiguous or overlapping tensor ranges: {shard}")
                    position = end
                if position != stat.st_size - base:
                    raise ValueError(f"Unindexed trailing bytes: {shard}")
                self.shards[shard] = {"base": base, "payload_bytes": position,
                                      "stat": (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)}
            if index and weight_map != index["weight_map"]:
                raise ValueError("Safetensors headers disagree with the checkpoint index")
            if unsupported:
                raise ValueError(f"Expert tensor layout needs an adapter: {unsupported[:3]}")
            expected = {(layer, expert) for layer in range(ranking.layers) for expert in range(ranking.experts)}
            if set(self.seats) != expected:
                raise ValueError(f"Incomplete expert geometry: missing={len(expected - self.seats.keys())}, "
                                 f"extra={len(self.seats.keys() - expected)}")
            self.suffixes = sorted(self.seats[(0, 0)])
            self.schema = {}
            offset = 0
            for suffix in self.suffixes:
                tensor = self.seats[(0, 0)][suffix]
                self.schema[suffix] = {"dtype": tensor.dtype, "shape": list(tensor.shape),
                                       "offset": offset, "nbytes": tensor.nbytes}
                offset += tensor.nbytes
            if not offset:
                raise ValueError("Empty expert record")
            self.seat_bytes = offset
            for seat, tensors in self.seats.items():
                if sorted(tensors) != self.suffixes:
                    raise ValueError(f"Nonuniform tensor set at seat {seat}")
                for suffix, tensor in tensors.items():
                    want = self.schema[suffix]
                    if (tensor.dtype, list(tensor.shape), tensor.nbytes) != (want["dtype"], want["shape"], want["nbytes"]):
                        raise ValueError(f"Nonuniform expert schema at {tensor.name}")
            self.tensor_manifest_sha256 = digest(canonical(manifest))
            self.provenance = {"model_path": str(self.model), "config_sha256": digest(config_bytes),
                               "index_metadata_total_size": index.get("metadata", {}).get("total_size") if index else None,
                               "actual_checkpoint_payload_bytes": sum(item["payload_bytes"] for item in self.shards.values())}
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.stack.close()

    def check_unchanged(self):
        for shard, source in self.files.items():
            stat = os.fstat(source.fileno())
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != self.shards[shard]["stat"]:
                raise ValueError(f"Checkpoint changed while reading: {shard}")

    def identity(self):
        """Per shard (device, inode, size, mtime_ns, ctime_ns), plus the tensor manifest. Any write to a shard
        changes its ctime, which cannot be set back without root."""
        shards = {}
        for name, source in sorted(self.files.items()):
            stat = os.fstat(source.fileno())
            shards[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        return {"model_path": str(Path(self.model).resolve()), "tensor_manifest_sha256": self.tensor_manifest_sha256,
                "shards": shards}

    def fingerprint(self, progress=None):
        """Bind the store to weight bytes, ignoring config/producer labels and header metadata.

        With fingerprint_cache_dir set, a full fingerprint computed earlier is reused while every shard keeps its
        file identity (the full SHA-256 of the 22.5 GB checkpoint takes about a minute per boot).
        APARETO_FULL_FINGERPRINT=1 forces the full hash and refreshes the cache."""
        self.check_unchanged()
        cache = None
        if self._fingerprint is None and self.fingerprint_cache_dir is not None:
            identity = self.identity()
            cache = Path(self.fingerprint_cache_dir) / f"fingerprint_{digest(canonical(identity))}.json"
            if cache.exists() and os.environ.get("APARETO_FULL_FINGERPRINT") != "1":
                saved = json.loads(cache.read_text())
                if saved.get("identity") != identity or saved["fingerprint"].get("tensor_manifest_sha256") != self.tensor_manifest_sha256:
                    raise ValueError(f"Fingerprint cache {cache} does not match this checkpoint's identity")
                self._fingerprint, self.fingerprint_source = saved["fingerprint"], f"identity_cache:{cache.name}"
        if self._fingerprint is None:
            shards = {}
            for name, info in self.shards.items():
                value = hashlib.sha256()
                for offset in range(0, info["payload_bytes"], 8 << 20):
                    value.update(read_at(self.files[name].fileno(), min(8 << 20, info["payload_bytes"] - offset),
                                         info["base"] + offset))
                shards[name] = {"payload_bytes": info["payload_bytes"], "sha256": value.hexdigest()}
                if progress:
                    progress(f"fingerprinted source shard {name}")
            self.check_unchanged()
            self._fingerprint = {"tensor_manifest_sha256": self.tensor_manifest_sha256, "shards": shards}
            self.fingerprint_source = "full_sha256"
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                partial = cache.with_suffix(f".{os.getpid()}.tmp")
                partial.write_text(json.dumps({"identity": identity, "fingerprint": self._fingerprint}, sort_keys=True))
                os.replace(partial, cache)
        return self._fingerprint

    def read_tensor(self, layer, expert, suffix):
        tensor = self.seats[(layer, expert)][suffix]
        return read_at(self.files[tensor.shard].fileno(), tensor.nbytes, tensor.offset)

    def read_record(self, layer, expert):
        result = bytearray(self.seat_bytes)
        for suffix in self.suffixes:
            field = self.schema[suffix]
            result[field["offset"]:field["offset"] + field["nbytes"]] = self.read_tensor(layer, expert, suffix)
        return result


def build_store(checkpoint, output, alignment=4096, progress=None):
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Store or partial output exists: {output}")
    stride = align_up(checkpoint.seat_bytes, alignment)
    ranking = checkpoint.ranking
    count = len(ranking.order)
    fingerprint = checkpoint.fingerprint(progress)
    header = {"version": VERSION, "magic": MAGIC.decode(), "layers": ranking.layers,
              "experts_per_layer": ranking.experts, "n_seats": count,
              "seat_bytes": checkpoint.seat_bytes, "record_stride": stride, "align": alignment,
              "tensor_schema": checkpoint.schema, "source_fingerprint": fingerprint,
              "provenance": checkpoint.provenance, "data_offset": 0,
              "index": [[int(seat) // ranking.experts, int(seat) % ranking.experts, 0, checkpoint.seat_bytes]
                        for seat in ranking.order],
              "record_sha256": ["0" * 64 for _ in range(count)],
              "payload_sha256": "0" * 64, "header_sha256": "0" * 64}
    for _ in range(16):
        length = len(canonical(header))
        offset = align_up(8 + length, alignment)
        if offset == header["data_offset"]:
            break
        header["data_offset"] = offset
        for rank, entry in enumerate(header["index"]):
            entry[2] = offset + rank * stride
    else:
        raise ValueError("Header offsets did not converge")
    total = header["data_offset"] + count * stride
    if total + (256 << 20) > shutil.disk_usage(output.parent).free:
        raise OSError(f"Insufficient space for {total} store bytes plus 256 MiB headroom")
    raw_header = canonical(header)
    if len(raw_header) > MAX_HEADER_BYTES:
        raise ValueError("Packed header exceeds the supported limit")
    payload = hashlib.sha256()
    padding = bytes(stride - checkpoint.seat_bytes)
    with partial.open("xb") as destination:
        destination.write(MAGIC + struct.pack("<I", len(raw_header)) + raw_header)
        destination.write(bytes(header["data_offset"] - destination.tell()))
        for rank, entry in enumerate(header["index"]):
            record = checkpoint.read_record(entry[0], entry[1])
            record.extend(padding)
            header["record_sha256"][rank] = digest(record)
            payload.update(record)
            destination.write(record)
            destination.flush()
            if progress and (rank == 0 or (rank + 1) % 512 == 0 or rank + 1 == count):
                progress(f"wrote {rank + 1}/{count} seats")
        checkpoint.check_unchanged()
        header["payload_sha256"] = payload.hexdigest()
        header["header_sha256"] = header_digest(header)
        final_header = canonical(header)
        if len(final_header) != len(raw_header) or destination.tell() != total:
            raise ValueError("Final store length/header differs from the allocation plan")
        destination.seek(8)
        destination.write(final_header)
        destination.flush()
        os.fsync(destination.fileno())
    # Publishing is exclusive even if another writer created output meanwhile.
    os.link(partial, output)
    partial.unlink()
    parent = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return header


class PackedStore:
    def __init__(self, path, ranking=None, checkpoint=None, verify_payload=True):
        self.path = Path(path)
        self.source = self.path.open("rb")
        self.ranking = ranking if ranking is not None else Ranking.expert_order()
        try:
            prefix = self.source.read(8)
            if len(prefix) != 8 or prefix[:4] != MAGIC:
                raise ValueError("Missing packed-store magic")
            length = struct.unpack("<I", prefix[4:])[0]
            if not 0 < length <= MAX_HEADER_BYTES:
                raise ValueError("Invalid packed-store header length")
            raw = self.source.read(length)
            if len(raw) != length:
                raise ValueError("Truncated packed-store header")
            self.header = decode_json(raw)
            h = self.header
            if h["version"] != VERSION or h["magic"] != MAGIC.decode():
                raise ValueError("Unsupported packed-store version")
            if h["header_sha256"] != header_digest(h):
                raise ValueError("Packed-store header checksum mismatch")
            if (h["layers"], h["experts_per_layer"], h["n_seats"]) != (
                    self.ranking.layers, self.ranking.experts, len(self.ranking.order)):
                raise ValueError("Packed-store geometry mismatch")
            self.seat_bytes = integer(h["seat_bytes"], "seat bytes", 1)
            self.stride = integer(h["record_stride"], "record stride", 1)
            if self.stride != align_up(self.seat_bytes, h["align"]):
                raise ValueError("Invalid record alignment/stride")
            self.data_offset = integer(h["data_offset"], "data offset", 8)
            if self.data_offset != align_up(8 + length, h["align"]):
                raise ValueError("Invalid data offset/header alignment")
            self.schema = h["tensor_schema"]
            offset = 0
            for suffix in sorted(self.schema):
                field = self.schema[suffix]
                if field["offset"] != offset or field["nbytes"] != tensor_size(field["dtype"], field["shape"]):
                    raise ValueError(f"Invalid tensor layout: {suffix}")
                offset += field["nbytes"]
            if offset != self.seat_bytes:
                raise ValueError("Tensor schema does not cover the seat payload")
            count = len(self.ranking.order)
            if len(h["index"]) != count or len(h["record_sha256"]) != count:
                raise ValueError("Incomplete packed-store index/checksums")
            for rank, seat in enumerate(self.ranking.order):
                expected = [int(seat) // self.ranking.experts, int(seat) % self.ranking.experts,
                            self.data_offset + rank * self.stride, self.seat_bytes]
                if h["index"][rank] != expected:
                    raise ValueError(f"Invalid packed-store index at rank {rank}")
            for value in [h["payload_sha256"], *h["record_sha256"]]:
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError("Invalid payload checksum encoding")
            if os.fstat(self.source.fileno()).st_size != self.data_offset + count * self.stride:
                raise ValueError("Packed-store file size disagrees with header/index")
            gap = read_at(self.source.fileno(), self.data_offset - 8 - length, 8 + length)
            if any(gap):
                raise ValueError("Nonzero header padding")
            if checkpoint is not None:
                if checkpoint.schema != self.schema or checkpoint.fingerprint() != h["source_fingerprint"]:
                    raise ValueError("Store does not match the source weight bytes/schema")
            self.inverse = np.empty(count, dtype=np.int64)
            self.inverse[self.ranking.order] = np.arange(count)
            self._padding = bytes(self.stride - self.seat_bytes)
            # Each record is checksummed before its first use in this process, not on every read:
            # per-read SHA-256 bounded streamed prefill at ~0.43 GB/s against ~6 GB/s for the read.
            # The file identity is rechecked on every read, so a replaced or rewritten file is refused.
            self._verified = np.zeros(count, dtype=bool)
            self._identity = self._file_identity()
            if verify_payload:
                self.verify_all()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.source.close()

    def rank_of(self, layer, expert):
        integer(layer, "layer")
        integer(expert, "expert")
        if layer >= self.ranking.layers or expert >= self.ranking.experts:
            raise ValueError("Seat outside store geometry")
        return int(self.inverse[layer * self.ranking.experts + expert])

    def _file_identity(self):
        stat = os.fstat(self.source.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def read_ranks(self, first, count):
        integer(first, "first rank")
        integer(count, "rank count")
        if first + count > len(self.ranking.order):
            raise ValueError("Read exceeds the ranked store")
        if self._file_identity() != self._identity:
            raise ValueError("Packed-store file changed after it was opened")
        data = read_at(self.source.fileno(), count * self.stride, self.data_offset + first * self.stride)
        view = memoryview(data)
        for index in range(count):
            if self._verified[first + index]:
                continue
            record = view[index * self.stride:(index + 1) * self.stride]
            if digest(record) != self.header["record_sha256"][first + index]:
                raise ValueError(f"Record checksum mismatch at rank {first + index}")
            if record[self.seat_bytes:] != self._padding:
                raise ValueError(f"Nonzero record padding at rank {first + index}")
            self._verified[first + index] = True
        return data

    def tensors(self, record):
        if len(record) not in (self.seat_bytes, self.stride):
            raise ValueError("Wrong record length")
        view = memoryview(record)
        return {suffix: view[field["offset"]:field["offset"] + field["nbytes"]]
                for suffix, field in self.schema.items()}

    def verify_all(self):
        value = hashlib.sha256()
        chunk = max(1, (8 << 20) // self.stride)
        for first in range(0, len(self.ranking.order), chunk):
            value.update(self.read_ranks(first, min(chunk, len(self.ranking.order) - first)))
        if value.hexdigest() != self.header["payload_sha256"]:
            raise ValueError("Packed-store payload checksum mismatch")

    def tier_ranges(self, pinned_seats, host_bytes):
        integer(pinned_seats, "pinned seats")
        integer(host_bytes, "host bytes")
        count = len(self.ranking.order)
        if pinned_seats > count:
            raise ValueError("Pinned tier exceeds the ranked store")
        cached = min(host_bytes // self.stride, count - pinned_seats)
        return {"vram": [0, pinned_seats], "host": [pinned_seats, pinned_seats + cached],
                "file": [pinned_seats + cached, count], "host_cache_seats": cached,
                "host_unused_bytes": host_bytes - cached * self.stride}
