"""Rollback-spacer read in the prompt's own forward: one pass over [prompt chunk | spacer], exact rollback.

The final prompt chunk is scheduled with the spacer appended, so the spacer costs no second prefill.
- Attention: causal, so prompt rows never see the spacer. The spacer's KV at positions P..P+len-1 is
  never read again (attention is bounded by sequence length) and decode overwrites it.
- Gated-DeltaNet: each layer runs the prompt rows, then the spacer rows, as two sequential kernel
  calls. The recurrent and conv state after the prompt rows is copied aside between them and copied
  back after the forward, so decode continues from the state at COMMIT.
- Sampling: the first token is sampled from the last prompt row, not the last spacer row.
Requires no speculative decoding, no prefix caching, the Triton GDN prefill
kernel and a scheduler lookahead of at least the spacer length.
"""
from dataclasses import replace

import torch
from vllm.forward_context import get_forward_context
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.v1.kv_cache_interface import MambaSpec


def _part(full, length, initial):
    """Single-sequence GDN prefill metadata for `length` rows of the same request slot."""
    device = full.non_spec_query_start_loc.device
    cpu = torch.tensor([0, length], dtype=full.non_spec_query_start_loc.dtype)
    location = cpu.to(device)
    nums_dict, batch_ptr, offsets = compute_causal_conv1d_metadata(cpu, device=device)
    has = full.has_initial_state if initial is None else torch.full_like(full.has_initial_state, initial)
    return replace(full, num_prefill_tokens=length, num_actual_tokens=length,
                   non_spec_query_start_loc=location, prefill_query_start_loc=location,
                   has_initial_state=has, prefill_has_initial_state=has,
                   chunk_indices=async_tensor_h2d(prepare_chunk_indices(cpu, FLA_CHUNK_SIZE), device=device),
                   chunk_offsets=async_tensor_h2d(prepare_chunk_offsets(cpu, FLA_CHUNK_SIZE), device=device),
                   nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets)


class SpacerRollback:
    def __init__(self, model, vllm_config, spacer):
        if not spacer or any(type(token) is not int or token < 0 for token in spacer):
            raise ValueError('Invalid spacer token IDs')
        # Async scheduling is fine: the split, the rollback and the sampling row are all inside the one prefill
        # forward, and the next step's input is the token sampled from the last prompt row.
        if vllm_config.speculative_config is not None:
            raise ValueError('Spacer rollback runs without speculative decoding')
        self.speculative = 0       # draft tokens per step: recurrent-state blocks per request are 1 + this
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('Spacer rollback requires prefix caching disabled')
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _resolve_gdn_prefill_backend
        if _resolve_gdn_prefill_backend(vllm_config)[1] != 'triton':
            raise ValueError('Spacer split was written against the Triton/FLA GDN prefill kernel')
        self.spacer = tuple(spacer)
        self.modules = {name: module for name, module in vllm_config.compilation_config.static_forward_context.items()
                        if isinstance(module, MambaBase)}
        if not self.modules:
            raise ValueError('Spacer rollback found no recurrent-state layers')
        # Scratch is allocated now, before KV sizing, so native memory accounting includes it.
        self.scratch = {}
        for index, (name, module) in enumerate(sorted(self.modules.items())):
            tensors = [torch.empty(shape, dtype=dtype, device=next(model.parameters()).device)
                       for shape, dtype in zip(module.get_state_shape(), module.get_state_dtype())]
            for part, tensor in enumerate(tensors):
                model.register_buffer(f'apareto_spacer_state_{index}_{part}', tensor)
            self.scratch[name] = tensors
            self._install(name, module)
        self.scratch_bytes = sum(t.nbytes for tensors in self.scratch.values() for t in tensors)
        self.groups = None
        self.active = None

    def _install(self, name, module):
        original = module._forward_core
        def split_core(mixed_qkv, b, a, core_attn_out):
            active = self.active
            metadata = get_forward_context().attn_metadata
            if active is None or metadata is None:
                return original(mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out)
            if not isinstance(metadata, dict):
                raise ValueError('Spacer split requires one attention-metadata dict per forward')
            full = metadata[module.prefix]
            rows, count = active['rows'], len(self.spacer)
            if (full.num_prefills, full.num_decodes, full.num_spec_decodes, full.num_actual_tokens) != (1, 0, 0, rows + count) \
                    or full.spec_sequence_masks is not None:
                raise ValueError('Spacer split requires exactly one prefill sequence of prompt rows plus spacer')
            key = id(full)
            if key not in active['parts']:
                active['parts'][key] = (full, _part(full, rows, None), _part(full, count, True))
            _, first, second = active['parts'][key]
            try:
                metadata[module.prefix] = first
                original(mixed_qkv=mixed_qkv[:rows], b=b[:rows], a=a[:rows], core_attn_out=core_attn_out[:rows])
                slot = self._slots(active)[name]
                for scratch, cache in zip(self.scratch[name], module.kv_cache):
                    scratch.copy_(cache[slot])
                metadata[module.prefix] = second
                original(mixed_qkv=mixed_qkv[rows:rows + count], b=b[rows:rows + count], a=a[rows:rows + count],
                         core_attn_out=core_attn_out[rows:rows + count])
            finally:
                metadata[module.prefix] = full
            active['split'].add(name)
        module._forward_core = split_core

    def _bind(self, runner):
        if self.groups is None:
            self.groups = {}
            for group, spec in enumerate(runner.kv_cache_config.kv_cache_groups):
                if isinstance(spec.kv_cache_spec, MambaSpec):
                    for name in spec.layer_names:
                        self.groups[name] = group
            if set(self.groups) != set(self.modules):
                raise ValueError('Recurrent-state layers differ from the KV cache groups')
        return self.groups

    def arm(self, runner, request_id, rows):
        """Split the next forward: `rows` prompt rows of this request, then the spacer."""
        if self.active is not None:
            raise ValueError('Spacer split already armed')
        for name, scratch in self.scratch.items():
            states = self.modules[name].kv_cache[:len(scratch)]
            if any(s.shape != c.shape[1:] or s.dtype != c.dtype for s, c in zip(scratch, states)):
                raise ValueError(f'Recurrent state layout differs at {name}')
        self._bind(runner)
        self.active = {'rows': rows, 'parts': {}, 'split': set(), 'runner': runner, 'request_id': request_id, 'slots': None}

    def _slots(self, active):
        # A new request is registered by the runner inside the step, so its slots are resolved at first use.
        if active['slots'] is None:
            request = active['runner'].requests[active['request_id']]
            active['slots'] = {name: request.block_ids[group][0] for name, group in self.groups.items()}
        return active['slots']

    def snapshot(self, runner, request_id):
        """Copy a decoding request's recurrent state aside (the refresh and post-think probes). Returns its slots;
        the caller checks the state is there."""
        if self.active is not None:
            raise ValueError('Spacer split is armed')
        self._bind(runner)
        request = runner.requests[request_id]
        slots = {}
        for name, group in self.groups.items():
            if len(request.block_ids[group]) != 1 + self.speculative:
                raise ValueError(f'Expected {1 + self.speculative} recurrent-state blocks per request at {name}')
            slots[name] = request.block_ids[group][0]
            for scratch, cache in zip(self.scratch[name], self.modules[name].kv_cache):
                scratch.copy_(cache[slots[name]])
        return slots

    def restore(self, slots):
        for name, slot in slots.items():
            for scratch, cache in zip(self.scratch[name], self.modules[name].kv_cache):
                cache[slot].copy_(scratch)

    def disarm(self):
        """Return the recurrent state to COMMIT; check every layer split."""
        active, self.active = self.active, None
        if active is None:
            raise ValueError('Spacer split was not armed')
        if active['split'] != set(self.modules):
            raise RuntimeError(f'Spacer split ran on {len(active["split"])} of {len(self.modules)} recurrent layers')
        for name, slot in self._slots(active).items():
            for scratch, cache in zip(self.scratch[name], self.modules[name].kv_cache):
                cache[slot].copy_(scratch)

