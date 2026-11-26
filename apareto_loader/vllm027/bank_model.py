"""Single-GPU expert-bank model; native dense/shared experts and router math.

Each MoE layer calls the bank engine with the router's own expert ids (0..255): the layer's slot table maps
them to bank slots inside the patched kernel. Prefill (a full step) runs every routed expert, loaded on
demand; decode masks routing to the request's keep-set, whose experts the bank holds (paged decode: the bank
holds some of them and the kernel reads the rest from pinned host memory).
"""
from contextlib import contextmanager
import os

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
from vllm.model_executor.models import qwen3_next
from vllm.model_executor.models.qwen3_5 import (Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeProcessingInfo, Qwen3VLMultiModalProcessor, Qwen3VLDummyInputsBuilder)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal import MULTIMODAL_REGISTRY

from apareto_loader.config import ExpertConfig, warm_set
from apareto_loader.store import Ranking
from .prefill import full_step


@contextmanager
def bank_factory(experts_class, runner_class):
    """Build the model's MoE layers with the bank's routed-experts and runner classes."""
    original = qwen3_next.FusedMoEFactory
    def factory(*args, **kwargs):
        return original(*args, **kwargs, routed_experts_cls=experts_class, runner_cls=runner_class)
    qwen3_next.FusedMoEFactory = factory
    try:
        yield
    finally:
        qwen3_next.FusedMoEFactory = original


class BankRoutedExperts(RoutedExperts):
    def __init__(self, layer_name, params_dtype, moe_config, quant_config, expert_map_manager, **kwargs):
        config = ExpertConfig.from_vllm(get_current_vllm_config())
        ranking = Ranking.expert_order()
        # Native quantization metadata uses an unreachable one-expert placeholder; routed computation
        # uses the bank.
        moe_config.num_local_experts = 1
        expert_map_manager._local_num_experts = 1
        super().__init__(layer_name, params_dtype, moe_config, quant_config,
                         expert_map_manager=expert_map_manager, **kwargs)
        self.apareto_layer = extract_layer_index(layer_name)
        # Decode mask, one row per request slot (one row when one request decodes). Starts as the warm set, which
        # the bank holds at boot.
        warm = warm_set(ranking.layers, ranking.experts, config.warm_experts)[self.apareto_layer]
        rows = get_current_vllm_config().scheduler_config.max_num_seqs
        self.register_buffer('apareto_keep', torch.tensor(warm, device=moe_config.device).repeat(rows, 1))


# Measurement only: APARETO_CPUX_NUMERICS=<new file>[:steps] with --enforce-eager compares,
# for every absent allowed (token, expert) of the first `steps` paged decode steps, today's GPU path (FP4
# activations, zero-copy), the CPU kernel (int8 activations) and an fp64 reference from the same record.
_NUMERICS = None
if os.environ.get('APARETO_CPUX_NUMERICS'):
    from .cpu_experts import NumericsProbe
    _NUMERICS = NumericsProbe(os.environ['APARETO_CPUX_NUMERICS'])


def allowed_rows(keep, row_slot, logical):
    """[tokens, 8] bool: each pick against its own token row's keep row. keep is
    [slots, 256]; row_slot[t] is row t's slot, -1 for a padding row, which allows nothing. One slot: keep[0][logical]."""
    if keep.shape[0] == 1:
        return keep[0][logical.long()]
    slot = row_slot[:logical.shape[0]].long()
    return keep[slot.clamp_min(0)[:, None], logical.long()] & (slot >= 0)[:, None]


class BankRunner(MoERunner):
    def _apply_quant_method(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        experts = self.routed_experts
        engine = experts.apareto_engine
        layer = experts.apareto_layer
        self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.NO_OVERLAP)
        weights, logical = self.router.select_experts(hidden_states, router_logits,
            self._quant_method.topk_indices_dtype, input_ids=input_ids)
        step = full_step()
        if step is not None and step.profiling:
            fused = engine.run(layer, hidden_states, logical.to(torch.int32).contiguous(), weights.contiguous(),
                               table=engine.profile_table)
        elif step is not None:
            engine.before_moe(layer, experts.apareto_features)
            fused = engine.run(layer, hidden_states, logical.to(torch.int32).contiguous(), weights.contiguous())
            engine.after_moe(layer)
        else:
            keep = experts.apareto_keep
            allowed = allowed_rows(keep, engine.row_slot, logical)
            ids = torch.where(allowed, logical, -1).to(torch.int32).contiguous()
            scales = torch.where(allowed, weights, 0).contiguous()
            absent = allowed & (engine.table[layer][logical.long()] < 0)
            if engine.paged and engine.cpu_experts is not None:
                # CPU-served misses: the ring kernel splits the routing; the GPU runs the
                # resident experts (and any absent one without a host record, zero-copy, as below); finish adds
                # the CPU's rows, or computes them zero-copy after the timeout.
                if _NUMERICS is not None:
                    _NUMERICS.observe(engine, layer, hidden_states, logical, weights, experts.apareto_keep[0])
                cpu = engine.cpu_experts
                gpu_ids, gpu_scales = cpu.ring(layer, logical.to(torch.int32).contiguous(), weights,
                                               keep, engine.table[layer], engine.host_table[layer, 0],
                                               hidden_states, engine.routed[layer],
                                               None if engine.rescue_w13 is None else engine.rescue_w13[layer],
                                               engine.row_slot)
                if engine.premask is not None:
                    # a served pick outside the keep-set is a rescue: on the CPU, or on the
                    # GPU when the bank holds it
                    rescued = ((cpu.served(logical.shape[0])[0] >= 0) | (gpu_ids >= 0)) & ~allowed
                    engine.record_premask(layer, logical, weights, allowed.to(torch.int8) + 2 * rescued.to(torch.int8))
                fused = engine.run(layer, hidden_states, gpu_ids, gpu_scales.to(weights.dtype), host=engine.host_table)
                cpu.finish(layer, fused)
            elif engine.paged:
                # Paged decode: an absent allowed expert is read from its host record.
                if _NUMERICS is not None:
                    _NUMERICS.observe(engine, layer, hidden_states, logical, weights, experts.apareto_keep[0])
                torch._assert_async(~(absent & (engine.host_table[layer, 0][logical.long()] == 0)).any(),
                                    'An allowed expert is absent from the bank and has no host record')
                engine.record_routed(layer, ids)
                if engine.premask is not None:
                    engine.record_premask(layer, logical, weights, allowed.to(torch.int8))
                fused = engine.run(layer, hidden_states, ids, scales, host=engine.host_table)
            else:
                torch._assert_async(~absent.any(), 'An allowed expert is absent from the bank')
                fused = engine.run(layer, hidden_states, ids, scales)
        self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
        self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
        return (self._shared_experts.output if self._shared_experts is not None else None, fused)


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder)
class BankQwen3_5Moe(Qwen3_5MoeForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix='model'):
        expert = ExpertConfig.from_vllm(vllm_config)
        ranking = Ranking.expert_order()
        text = vllm_config.model_config.hf_text_config
        if (text.num_hidden_layers, text.num_experts) != (ranking.layers, ranking.experts):
            raise ValueError('Checkpoint geometry does not match 40 layers x 256 experts')
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1 or parallel.enable_eplb:
            raise ValueError('Bank serving requires one GPU and no expert load balancing')
        # Async scheduling is allowed (the host prepares step n+1 while the GPU runs step n): the worker learns a
        # step's tokens one step later, which the </think> swap allows for (bank_worker.execute_model).
        if vllm_config.speculative_config is not None:
            raise ValueError('Bank serving runs without speculative decoding')
        if vllm_config.scheduler_config.max_num_seqs != 1:
            # Concurrent decode: paged decode; sync or async scheduling.
            if expert.gpu_experts is None:
                raise ValueError('Concurrent decode (max_num_seqs > 1) needs paged decode (--gpu-experts)')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('Bank serving requires --no-enable-prefix-caching: taps are read from computed prompt rows')
        with bank_factory(BankRoutedExperts, BankRunner):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
