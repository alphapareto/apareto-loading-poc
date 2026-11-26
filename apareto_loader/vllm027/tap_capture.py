"""Graph-visible copies of final prompt rows, before attention mutates K/V."""
import numpy as np
import torch
import triton
import triton.language as tl

from apareto_loader.selector import SelectionRejected
from apareto_loader.taps import DIMS, FEATURES, PromptWindow, TAPS


@triton.jit
def _copy_rows(source, destination, seen, indices, srcbad, stride_row: tl.constexpr,
               stride_col: tl.constexpr, source_rows, width: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = offset // width, offset % width
    local = tl.load(indices + row, row < 8, other=-1)
    valid = (row < 8) & (local >= 0) & (local < source_rows)
    value = tl.load(source + local * stride_row + column * stride_col, valid, other=0)
    tl.store(destination + offset, value, valid)
    tl.store(seen + row, 1, valid & (column == 0))
    # A NaN fails its own equality test. Flags rows whose SOURCE was already non-finite, which is
    # what distinguishes a broken forward from a bad copy; costs one comparison per element.
    tl.store(srcbad + row, 1, valid & (value != value))


@torch.library.custom_op("apareto::capture_tap_rows", mutates_args=("destination", "seen", "srcbad"))
def capture_rows(source: torch.Tensor, destination: torch.Tensor, seen: torch.Tensor,
                 indices: torch.Tensor, srcbad: torch.Tensor) -> None:
    width = destination.shape[1]
    _copy_rows[(triton.cdiv(destination.numel(), 256),)](
        source, destination, seen, indices, srcbad, source.stride(0), source.stride(1),
        source.shape[0], width, 256)


@capture_rows.register_fake
def _capture_rows_fake(source, destination, seen, indices, srcbad):
    return None


class TapCapture(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer("features", torch.full((FEATURES,), float('nan'), dtype=torch.float16, device=device))
        self.register_buffer("indices", torch.full((8,), -1, dtype=torch.int64, device=device))
        self.register_buffer("seen", torch.zeros((len(TAPS), 8), dtype=torch.bool, device=device))
        self.register_buffer("srcbad", torch.zeros((len(TAPS), 8), dtype=torch.bool, device=device))
        self.views = []
        offset = 0
        for width in DIMS:
            self.views.append(self.features[offset:offset + 8 * width].view(8, width))
            offset += 8 * width
        self.window = None
        self.hooks = []

    def capture(self, tap, source):
        index = TAPS.index(tap)
        capture_rows(source, self.views[index], self.seen[index], self.indices, self.srcbad[index])

    def attach(self, model):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
        layers = {module.layer_idx: module for module in model.modules() if isinstance(module, Qwen3_5DecoderLayer)}
        if len(layers) != 40:
            raise ValueError("Tap capture requires the 40 native decoder layers")
        for tap, width in zip(TAPS, DIMS):
            if tap.startswith('hl'):
                layer = layers[int(tap[2:])]
                def residual_hook(module, args, output, name=tap):
                    # Fused add+norm returns the summed, unnormalized residual as output[1].
                    # Copy it now before later fused operations reuse its storage.
                    residual = output[1] if isinstance(output, tuple) else args[0]
                    self.capture(name, residual)
                self.hooks.append(layer.input_layernorm.register_forward_hook(residual_hook))
            else:
                layer = layers[int(tap[3:])]
                if layer.layer_type != 'full_attention' or 2 * layer.self_attn.kv_size != width:
                    raise ValueError(f"Wrong native QKV tap geometry at {tap}")
                def kv_hook(module, args, output, name=tap, columns=width):
                    self.capture(name, output[0][:, -columns:])
                self.hooks.append(layer.self_attn.qkv_proj.register_forward_hook(kv_hook))

    def begin(self, request_id, prompt_tokens, first=0):
        """Read the last 8 of positions [0, prompt_tokens), fed in chunks from position `first` on (0 for a
        prompt; the probe's first position for the post-think answer probe)."""
        if self.window is not None:
            raise ValueError("Previous prompt capture has not been released")
        self.window = PromptWindow(request_id, prompt_tokens, first)
        self.features.fill_(float('nan'))
        self.seen.zero_(); self.srcbad.zero_()
        self.indices.fill_(-1)

    def prepare_step(self, start, count):
        if self.window is None:
            raise ValueError("No active prompt capture")
        indices = self.window.indices(start, count)
        self.indices.copy_(torch.from_numpy(indices))

    def disable(self):
        self.indices.fill_(-1)

    def collect(self):
        if self.window is None or not self.window.complete or not bool(self.seen.all().item()):
            raise ValueError("Missing prompt rows or taps")
        result = self.features.cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise SelectionRejected("Captured prompt contains nonfinite features")
        return result

    def release(self):
        self.disable()
        self.window = None
