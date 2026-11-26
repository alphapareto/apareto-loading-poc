"""Bounded host preparation for the synchronous full-prefill NVFP4 control.

Only permutations happen on the CPU. Floating-point scale arithmetic remains on
the GPU, in the same order as the native FlashInfer preparation path.
"""
from dataclasses import dataclass

import torch

from apareto_loader.store import EXPERT_SCALE_CONVENTION, expert_format
from .nvfp4 import PreparedWeights


def swizzle_host(scale):
    """Byte-equivalent to vLLM's swizzle_blockscale, without its implicit .cuda()."""
    if scale.device.type != "cpu" or scale.dtype != torch.float8_e4m3fn or scale.ndim != 3:
        raise ValueError("Expected a three-dimensional host FP8 scale stack")
    count, rows, columns = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = torch.zeros((count, padded_rows, padded_columns), dtype=torch.uint8)
    padded[:, :rows, :columns].copy_(scale.view(torch.uint8))
    return padded.reshape(count, padded_rows // 128, 4, 32, padded_columns // 4, 4).permute(
        0, 1, 4, 3, 2, 5).contiguous().reshape(count, padded_rows, padded_columns).view(torch.float8_e4m3fn)


@dataclass
class HostWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor
    s2: torch.Tensor
    global13: torch.Tensor
    global2: torch.Tensor
    # Checkpoint-native global scales are kept verbatim so the layer store round-trips byte-exactly
    # (float32 reciprocals are not involutive -- 12 of 72 sampled modelopt scales move by 1 ULP).
    # The convention is therefore applied once, at the kernel boundary in copy_to.
    scale_convention: str = "divide"

    @classmethod
    def from_raw(cls, w13, w2, s13, s2, global13, global2, scale_convention="divide"):
        if any(t.device.type != "cpu" for t in (w13, w2, s13, s2, global13, global2)):
            raise ValueError("Full-prefill preparation requires host tensors")
        middle = w13.shape[1] // 2
        if middle % 128 or w2.shape[1] % 128 or w13.shape[2] % 32 or w2.shape[2] % 32:
            raise ValueError("This NVFP4 stream adapter needs the validated aligned tensor layout")
        # Native FlashInfer CUTLASS consumes [up, gate], while checkpoint input is [gate, up].
        weights = torch.cat((w13[:, middle:], w13[:, :middle]), dim=1)
        scales = torch.cat((s13[:, middle:].view(torch.uint8), s13[:, :middle].view(torch.uint8)), dim=1)
        return cls(weights, w2, swizzle_host(scales.view(torch.float8_e4m3fn)), swizzle_host(s2),
                   global13[:, 0].contiguous(), global2.contiguous(), scale_convention)

    @classmethod
    def from_records(cls, records, schema):
        """Prepare only the layer's non-pinned records; callers decide their source tier."""
        count = len(records)
        if not count:
            raise ValueError("No non-pinned records to prepare")
        names = expert_format(schema)[1]
        middle, packed_hidden = schema[f"gate_proj.{names['packed']}"]["shape"]
        hidden = packed_hidden * 2
        raw = {
            "w13": torch.empty((count, 2 * middle, packed_hidden), dtype=torch.uint8),
            "w2": torch.empty((count, hidden, middle // 2), dtype=torch.uint8),
            "s13": torch.empty((count, 2 * middle, hidden // 16), dtype=torch.float8_e4m3fn),
            "s2": torch.empty((count, hidden, middle // 16), dtype=torch.float8_e4m3fn),
            "global13": torch.empty((count, 2), dtype=torch.float32),
            "global2": torch.empty(count, dtype=torch.float32),
        }
        for index, fields in enumerate(records):
            for projection, target, part in (("gate_proj", "13", 0), ("up_proj", "13", 1),
                                              ("down_proj", "2", 0)):
                for role, name, dtype in (("packed", "w", torch.uint8),
                                          ("block_scale", "s", torch.float8_e4m3fn),
                                          ("global_scale", "global", torch.float32)):
                    suffix = names[role]
                    source = torch.frombuffer(fields[projection + "." + suffix], dtype=dtype)
                    destination = raw[name + target][index]
                    if target == "13":
                        destination = destination[part] if name == "global" else destination[part * middle:(part + 1) * middle]
                    destination.copy_(source.reshape(destination.shape))
        return cls.from_raw(**raw, scale_convention=EXPERT_SCALE_CONVENTION[expert_format(schema)[0]])

    def copy_to(self, destination, logical_ids, input_dequant13, input_dequant2):
        """Copy into preallocated full-layer slots; no transient GPU weight stack."""
        if len(logical_ids) != len(self.w13):
            raise ValueError("Host records and destination expert IDs differ")
        for index, logical in enumerate(logical_ids):
            for name in ("w13", "w2", "s13", "s2"):
                getattr(destination, name)[logical].copy_(getattr(self, name)[index])
        device = destination.w13.device
        indices = torch.tensor(logical_ids, dtype=torch.int64, device=device)
        # Match native reciprocal followed by FP32 multiply, on the same device. compressed-tensors'
        # weight_global_scale DIVIDES, so it is reciprocated here; modelopt's weight_scale_2 is already
        # the multiplier and must be passed straight through. Reciprocating it anyway does not fail --
        # it scales every expert by ~1e9 and still produces output.
        if self.scale_convention == "divide":
            alpha13, alpha2 = 1.0 / self.global13.to(device), 1.0 / self.global2.to(device)
        elif self.scale_convention == "multiply":
            alpha13, alpha2 = self.global13.to(device), self.global2.to(device)
        else:
            raise ValueError(f"Unknown global-scale convention: {self.scale_convention!r}")
        destination.alpha13.index_copy_(0, indices, alpha13 * input_dequant13)
        destination.alpha2.index_copy_(0, indices, alpha2 * input_dequant2)


def allocate_layer(experts, hidden, middle, device):
    """One reusable kernel-ready layer, allocated before native KV profiling."""
    return PreparedWeights(
        torch.empty((experts, 2 * middle, hidden // 2), dtype=torch.uint8, device=device),
        torch.empty((experts, hidden, middle // 2), dtype=torch.uint8, device=device),
        torch.empty((experts, 2 * middle, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((experts, hidden, middle // 16), dtype=torch.float8_e4m3fn, device=device),
        *(torch.empty(experts, dtype=torch.float32, device=device) for _ in range(4)))


def copy_pinned(destination, module, logical_ids):
    """Reuse already prepared VRAM experts without reading their file records."""
    for slot, logical in enumerate(logical_ids):
        for target, source in (("w13", "w13_weight"), ("w2", "w2_weight"),
                               ("s13", "w13_weight_scale"), ("s2", "w2_weight_scale"),
                               ("alpha13", "w13_weight_scale_2"), ("alpha2", "w2_weight_scale_2")):
            getattr(destination, target)[logical].copy_(getattr(module, source)[slot])
    destination.a13_quant.copy_((1.0 / module.w13_input_scale[0]).expand_as(destination.a13_quant))
    destination.a2_quant.copy_((1.0 / module.w2_input_scale[0]).expand_as(destination.a2_quant))
