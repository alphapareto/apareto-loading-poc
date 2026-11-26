"""Fixed-stack NVFP4 preparation and dispatch with baseline FlashInfer math.

Dynamic shared-pool indirection is a separate adapter.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class PreparedWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor
    s2: torch.Tensor
    a13_quant: torch.Tensor
    a2_quant: torch.Tensor
    alpha13: torch.Tensor
    alpha2: torch.Tensor

    @classmethod
    def from_raw(cls, w13, w2, s13, s2, global13, global2, input13, input2):
        """Native transformations; callers supply full-layer input-scale constants."""
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import prepare_nvfp4_moe_layer_for_fi_or_cutlass
        layer = SimpleNamespace(activation=MoEActivation.SILU)
        w13, s13, g13, a13, w2, s2, g2, a2 = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
            NvFp4MoeBackend.FLASHINFER_CUTLASS, layer, w13, s13, 1.0 / global13[:, 0],
            1.0 / input13, w2, s2, 1.0 / global2, 1.0 / input2, True)
        # FlashInferExperts.process_weights_after_loading fuses these same factors.
        return cls(w13, w2, s13, s2, 1.0 / a13, 1.0 / a2, g13 * a13, g2 * a2)

    def select(self, slots):
        # Used for probes only. Serving allocates compact raw stacks before loading.
        selected = {}
        for name in self.__dataclass_fields__:
            tensor = getattr(self, name)
            raw = tensor.view(torch.uint8).reshape(tensor.shape[0], -1)[slots].contiguous()
            selected[name] = raw.view(tensor.dtype).reshape(len(slots), *tensor.shape[1:])
        return PreparedWeights(**selected)

    def workspace_bytes(self, tokens, top_k):
        from flashinfer.fused_moe.core import cutlass_fused_moe_workspace_size
        return cutlass_fused_moe_workspace_size(tokens, self.w2.shape[1], self.w2.shape[2] * 2,
            self.w2.shape[0], top_k, x_dtype=torch.uint8, weight_dtype=torch.int64,
            output_dtype=torch.bfloat16, device=self.w2.device)

    def run(self, hidden, ids, weights, workspace, output=None):
        from flashinfer.fused_moe.core import cutlass_fused_moe
        from vllm import _custom_ops as ops
        packed, input_sf = ops.scaled_fp4_quant(hidden, self.a13_quant)
        if output is None:
            output = torch.empty_like(hidden)
        cutlass_fused_moe(input=packed, token_selected_experts=ids,
            token_final_scales=weights, fc1_expert_weights=self.w13.view(torch.int64),
            fc2_expert_weights=self.w2.view(torch.int64), output_dtype=torch.bfloat16,
            quant_scales=[self.a13_quant, self.s13.view(torch.int32), self.alpha13,
                          self.a2_quant, self.s2.view(torch.int32), self.alpha2],
            input_sf=input_sf, output=output, workspace_buffer=workspace)
        return output


def mask_and_remap(logical_ids, weights, keep, seat_to_slot):
    """Keep membership is independent of residency; no rerouting or renormalization."""
    allowed = keep[logical_ids.long()]
    slots = seat_to_slot[logical_ids.long()]
    missing = allowed & (slots < 0)
    ids = torch.where(allowed, slots, -1).to(torch.int32)
    scales = torch.where(allowed, weights, 0)
    # A tensor flag is consumed by the worker audit; avoid a synchronizing .item() in graphs.
    return ids.contiguous(), scales.contiguous(), missing
