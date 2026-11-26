"""Bank-engine model loading: native dense weights, then the expert bank.

Order:
1. Native dense weights (attention, GDN, shared experts, embeddings). Routed experts are never loaded natively,
   nor read from the checkpoint (skipped by name in _get_weights_iterator).
2. The bank, sized for the budget (bank.bank_minimum) or to gpu_experts (paged decode), and
   allocated before vLLM's memory profiling so its KV sizing accounts for it. The fit check is in exact bytes:
   free memory now >= bank + KV + activation reserve + margin (no utilization fraction).
3. The host RAM tier: a bounded LRU of layer-store records in pinned RAM (host_tier.py says why bounded).
4. The patched MoE kernel and early R* are bound.
5. Warm start: the warm set (config.warm_set: the first experts of every layer) is loaded and is every layer's
   decode mask until the first request selects, so every decode table entry the router can reach is valid from the
   first step (and during CUDA-graph capture).
"""
from dataclasses import replace
import time

import numpy as np
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from apareto_loader import paths
from apareto_loader.bank import bank_minimum
from apareto_loader.config import ExpertConfig, cpu_experts_spec, cpu_rescue_spec, warm_set
from apareto_loader.layer_store import LayerStore, check_source_header
from apareto_loader.store import Checkpoint, EXPERT_RE, Ranking, expert_format
from . import moe_slots
from .bank_engine import BankEngine, early_selector
from .bank_model import BankRoutedExperts
from .host_tier import BankLoader, HostTier


class BankModelLoader(DefaultModelLoader):
    def __init__(self, load_config):
        super().__init__(replace(load_config, load_format='safetensors'))

    def _get_weights_iterator(self, source):
        """vLLM's default safetensors path, except that routed-expert tensors are skipped by name before they
        are read: the bank serves experts from the layer store. vLLM's own EP filter only skips names ending in
        `.weight`, and this checkpoint stores experts as weight_packed / weight_scale / *_global_scale."""
        from safetensors import safe_open
        from vllm.model_executor.model_loader.weight_utils import _natural_sort_key
        _, files, use_safetensors = self._prepare_weights(source.model_or_path, source.subfolder, source.revision,
                                                          source.fall_back_to_pt, source.allow_patterns_overrides)
        extra = self.load_config.model_loader_extra_config or {}
        if not use_safetensors or self.load_config.safetensors_load_strategy is not None or extra.get('enable_multithread_load'):
            raise ValueError('The bank loader reads safetensors with the default (lazy) strategy only')
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        for path in sorted(files, key=_natural_sort_key):
            with safe_open(path, framework='pt') as f:
                for name in f.keys():  # noqa: SIM118
                    if EXPERT_RE.fullmatch(name) is None:
                        yield source.prefix + name, f.get_tensor(name)

    def load_weights(self, model, model_config):
        if not any(isinstance(m, BankRoutedExperts) for m in model.modules()):
            # A model without bank layers loads as vLLM's default loader would.
            return super().load_weights(model, model_config)
        native = get_current_vllm_config()
        config = ExpertConfig.from_vllm(native)
        ranking = Ranking.expert_order()
        self._init_ep_weight_filter(model_config)
        model.load_weights((name, tensor) for name, tensor in self.get_all_weights(model_config, model)
                           if EXPERT_RE.fullmatch(name) is None)
        self._attach_bank(model, model_config, native, config, ranking)

    def _attach_bank(self, model, model_config, native, config, ranking):
        modules = {m.apareto_layer: m for m in model.modules() if isinstance(m, BankRoutedExperts)}
        if set(modules) != set(range(ranking.layers)):
            raise ValueError('Bank model layers do not match 40 x 256')
        with Checkpoint(model_config.model, ranking) as checkpoint:
            checkpoint.fingerprint_cache_dir = paths.cache() / 'apareto' / 'fingerprints'
            # config.store: the source header the layer store was built from (the store itself is deleted).
            source = check_source_header(config.store, ranking, checkpoint)
            input_suffix = expert_format(checkpoint.schema)[1]['input_scale']
            constants = []
            with torch.no_grad():
                for layer, module in sorted(modules.items()):
                    for name, parameter in module.named_parameters():
                        parameter.fill_(0 if 'packed' in name else 1)   # unreachable native placeholder
                    scales = {projection: np.array([np.frombuffer(checkpoint.read_tensor(layer, expert,
                        projection + '.' + input_suffix), dtype='<f4')[0] for expert in range(ranking.experts)])
                              for projection in ('gate_proj', 'up_proj', 'down_proj')}
                    constants.append([float(min(scales['gate_proj'].min(), scales['up_proj'].min())),
                                      float(scales['down_proj'].min())])
            store = LayerStore(config.layer_store, ranking, source['header_sha256'])
            device = next(model.parameters()).device
            predictor = early_selector(config.think_bundle)
            # Before early R* predicts, the bank keeps only the decode set (pre_prediction_keep 'decode').
            minimum = bank_minimum(config.budget, predictor.layer_cap, 'decode', ranking.layers, ranking.experts)
            slots = minimum
            # Paged decode: gpu_experts below the minimum; D is then partly host-served.
            # Concurrent decode and the CPU experts (the rescue) run on the paged engine even when the bank could hold
            # one request's whole decode set: the bank then holds the union of the live sets as far as it fits.
            rows = native.scheduler_config.max_num_seqs
            paged = config.gpu_experts is not None and (config.gpu_experts < minimum or rows > 1
                                                        or config.cpu_experts is not None)
            if config.gpu_experts is not None:
                slots = config.gpu_experts
            if slots < minimum and not paged:
                raise ValueError(f'Bank of {slots} slots is below the {minimum} that budget {config.budget} needs')
            free, total = torch.cuda.mem_get_info(device)
            bank_bytes = slots * store.header['record_bytes']
            kv = native.cache_config.kv_cache_memory_bytes
            if kv is None:
                raise ValueError('The bank engine sizes GPU memory in exact bytes: pass --kv-cache-memory-bytes')
            need = bank_bytes + kv + config.activation_reserve_bytes + config.gpu_margin_bytes
            if need > free:
                raise ValueError(f'Bank of {slots} slots ({bank_bytes} B) + KV {kv} B + activation reserve '
                                 f'{config.activation_reserve_bytes} B + margin {config.gpu_margin_bytes} B = {need} B '
                                 f'exceeds the {free} B free after the dense weights')
            engine = BankEngine(slots, store.layout, constants,
                                native.scheduler_config.max_num_batched_tokens, device,
                                rows=native.scheduler_config.max_num_seqs)
            engine.pre_prediction_keep = 'decode'
            model.add_module('apareto_engine', engine)
            for module in modules.values():
                object.__setattr__(module, 'apareto_engine', engine)
                object.__setattr__(module, 'apareto_features', None)   # bound by the worker to its taps
            # A host tier at or above the store's size is the whole store (preloaded), so a round number such as
            # 17 GiB for the 16.91 GiB store means 'everything in RAM'.
            capacity = (None if config.host_tier_bytes is None
                        else min(config.host_tier_bytes // store.stride, ranking.layers * ranking.experts))
            host = HostTier(config.layer_store, ranking, source['header_sha256'], capacity=capacity)
            loader = BankLoader(host, engine.fields, device=device)
            store.close()
            engine.bind(loader, predictor, moe_slots.load())
            warm = warm_set(ranking.layers, ranking.experts, config.warm_experts)
            if paged:
                # A tier that holds the whole store never evicts: prefill's layer pair and the staged records are
                # the same records as the decode set's, already resident. A smaller tier needs room for all three.
                whole = host.capacity == ranking.layers * ranking.experts
                if not whole and host.capacity < config.budget + 2 * ranking.experts + loader.staging_records:
                    raise ValueError(f'Paged decode holds the decode set in the host tier: {host.capacity} records '
                                     f'cannot hold {config.budget} next to a prefill layer pair')
                engine.enable_paged(host.store.layout)
                if config.cpu_experts is not None:
                    engine.enable_cpu_experts(cpu_experts_spec(config.cpu_experts),
                                              None if config.cpu_rescue is None else cpu_rescue_spec(config.cpu_rescue))
                loaded, events = engine.prepare_paged(warm)
            elif warm.sum() > slots:
                raise ValueError('The warm set does not fit in the bank')
            else:
                loaded, events = engine.prepare_decode(warm)
            torch.cuda.synchronize()
            engine.bank.unprotect_all()
            model.apareto_host = host
            model.apareto_load_report = {
                'engine': 'bank', 'bank_slots': slots, 'bank_minimum_slots': minimum, 'bank_bytes': bank_bytes,
                'gpu_experts': config.gpu_experts, 'paged_decode': paged,
                'cpu_experts': config.cpu_experts, 'cpu_rescue': config.cpu_rescue,
                'cpu_isolated': getattr(engine.cpu_experts, 'isolated', None),
                'pre_prediction_keep': 'decode', 'gpu_free_after_dense_bytes': free,
                'gpu_spare_at_load_bytes': free - need,
                'record_bytes': host.store.header['record_bytes'], 'layer_store_header_sha256': host.store.header['header_sha256'],
                'layer_store_record_order': host.store.record_order, 'host_tier_bytes': host.size, 'host_tier_records': host.capacity,
                'host_tier_after_warm_start': dict(host.stats),
                'host_tier_boot_seconds': host.boot_seconds, 'warm_set_experts': loaded,
                'full_layer_input_scales': constants,
                'moe_module': moe_slots.build_spec().name,
                'checkpoint_fingerprint_source': checkpoint.fingerprint_source}
            checkpoint.check_unchanged()
