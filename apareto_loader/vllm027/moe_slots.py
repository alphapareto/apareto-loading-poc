"""FlashInfer CUTLASS fused MoE with a per-call expert -> bank-slot table.

The stock kernel locates expert e's weights and weight block scales as `base + e × size`, so it accepts only
a contiguous block of experts, and its per-expert setup cost grows with that block. This patch adds an
optional 7th NVFP4 quant tensor, `expert_slot [num_experts]` (int32). When present:
- the kernel is told `num_experts = len(expert_slot)` (a layer's 256 experts);
- the weights and block scales of expert e are read from slot `expert_slot[e]` of the (larger) bank;
- per-expert alphas and activation scales stay indexed by e: the caller passes per-layer [256] arrays.
Everything else is the stock kernel. The table's address is fixed per call site, so CUDA graphs replay it
with whatever contents it holds.

Host experts (the paged-decode miss path): an optional 8th tensor, `expert_host [4, num_experts]`
(int64), holds device-visible addresses in pinned host memory of each expert's fc1 weights, fc1 block scales,
fc2 weights and fc2 block scales (rows in that order). An expert whose slot is -1 and whose tokens are > 0 is
then read by the TMA grouped GEMM straight from those addresses (TMA reads mapped host memory). Its alphas stay indexed by expert, so the caller sets
them for host experts too. Without the 8th tensor nothing changes.

Built as our own JIT module (`fused_moe_120_apareto_slots`) from patched copies of three stock files. Each
edit is an exact-match replacement with an expected count, so a FlashInfer change fails the build instead of
compiling something different. Slot mode refuses the non-TMA fallback GEMM path, which this patch does not
cover.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path

MODULE = 'fused_moe_120_apareto_slots'
BACKEND = 'fused_moe/cutlass_backend'
HEADER = 'nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h'

# (file relative to FLASHINFER_CSRC_DIR, old, new, expected occurrences)
EDITS = [
    # QuantParams gains the slot table (all three users of this header are compiled from the overlay).
    (HEADER,
     '''    GemmInputs fc1;
    GemmInputs fc2;
  } fp4;''',
     '''    GemmInputs fc1;
    GemmInputs fc2;
    int const* expert_slot = nullptr;  // apareto: (num_experts_per_node,) expert -> weight slot
  } fp4;''', 1),
    # Block scales: weight offset by slot, activation offset by expert.
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale,
    int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_act);''',
     '''    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale,
    int64_t num_tokens_before_expert, int64_t weight_index) {
  assert(layout_info.fpX_block_scaling_factors_stride_act);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''      weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);''',
     '''      weight_block_scale + getOffsetWeightSF(weight_index, gemm_n, gemm_k, scaling_type);''', 1),
    # Weights: pointer by slot.
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''    int const* permuted_row_to_unpermuted_row, int64_t const out_idx) {''',
     '''    int const* permuted_row_to_unpermuted_row, int64_t const out_idx, int64_t const weight_index) {''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''  layout_info.ptr_weight[out_idx] = safe_inc_ptr(weights, expert * (gemm_n * gemm_k));''',
     '''  layout_info.ptr_weight[out_idx] = safe_inc_ptr(weights, weight_index * (gemm_n * gemm_k));''', 1),
    # The setup kernel resolves the slot once per active expert.
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;''',
     '''  int64_t const weight_index =
      quant_params.fp4.expert_slot ? int64_t(quant_params.fp4.expert_slot[expert]) : int64_t(expert);
  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''          quant_type.fc1.weight_block_scale, num_tokens_before_expert);''',
     '''          quant_type.fc1.weight_block_scale, num_tokens_before_expert, weight_index);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''          quant_type.fc2.weight_block_scale, num_tokens_before_expert);''',
     '''          quant_type.fc2.weight_block_scale, num_tokens_before_expert, weight_index);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''      bias1, gemm1_output, nullptr, nullptr, expert);''',
     '''      bias1, gemm1_output, nullptr, nullptr, expert, weight_index);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''      bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, expert);''',
     '''      bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, expert, weight_index);''', 1),
    # Slot mode is only implemented for the TMA warp-specialized grouped GEMMs.
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''  TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
  TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");''',
     '''  TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
  TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");
  TLLM_CHECK_WITH_INFO(!quant_params.fp4.expert_slot ||
                           (moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
                            moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)),
                       "apareto expert_slot requires TMA warp-specialized MoE GEMMs");''', 1),
    # Binding: the expert count comes from the slot table when one is given.
    (f'{BACKEND}/flashinfer_cutlass_fused_moe_binding.cu',
     '''    int const num_experts_on_rank = fc2_expert_weights.size(0);
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);''',
     '''    int const num_experts_on_rank = apareto_slot_experts(quant_scales, fc2_expert_weights.size(0));
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);''', 2),
    (f'{BACKEND}/flashinfer_cutlass_fused_moe_binding.cu',
     '''  kernels::QuantParams getQuantParams(
      int64_t num_experts_on_rank, int64_t hidden_size, int64_t inter_size,''',
     '''  // apareto: an optional 7th NVFP4 quant tensor is an int32 expert -> weight slot table.
  int64_t apareto_slot_experts(Optional<Array<Tensor>> const& quant_scales, int64_t weight_experts) const {
    if (isNvfp4Quant() && quant_scales.has_value() && quant_scales.value().size() == 7) {
      auto const& slots = quant_scales.value()[6];
      CHECK_INPUT_TYPE(slots, dl_int32);
      CHECK_DIM(1, slots);
      return slots.size(0);
    }
    return weight_experts;
  }

  kernels::QuantParams getQuantParams(
      int64_t num_experts_on_rank, int64_t hidden_size, int64_t inter_size,''', 1),
    (f'{BACKEND}/flashinfer_cutlass_fused_moe_binding.cu',
     '''      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 6)
          << "Expecting 6 quant scales for nvfp4 quantization";''',
     '''      TVM_FFI_ICHECK(quant_scales.value().size() == 6 || quant_scales.value().size() == 7)
          << "Expecting 6 quant scales (7 with an apareto slot table) for nvfp4 quantization";
      bool const apareto_slots = quant_scales.value().size() == 7;''', 1),
    # --- host experts (8th tensor) ---
    (HEADER,
     '''    int const* expert_slot = nullptr;  // apareto: (num_experts_per_node,) expert -> weight slot''',
     '''    int const* expert_slot = nullptr;  // apareto: (num_experts_per_node,) expert -> weight slot
    int64_t const* expert_host = nullptr;  // apareto: (4, num_experts_per_node) host addresses, slot -1 experts''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''  int64_t const weight_index =
      quant_params.fp4.expert_slot ? int64_t(quant_params.fp4.expert_slot[expert]) : int64_t(expert);''',
     '''  int64_t const weight_index =
      quant_params.fp4.expert_slot ? int64_t(quant_params.fp4.expert_slot[expert]) : int64_t(expert);
  // apareto: an absent expert (slot -1) with a host-address table is read from pinned host memory, at index 0
  // of its own record's fields (getOffsetWeightSF(0, ...) == 0).
  bool const apareto_host = quant_params.fp4.expert_slot && quant_params.fp4.expert_host && weight_index < 0;
  int64_t const* const apareto_addr = quant_params.fp4.expert_host;''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''          quant_type.fc1.weight_block_scale, num_tokens_before_expert, weight_index);''',
     '''          apareto_host ? reinterpret_cast<decltype(quant_type.fc1.weight_block_scale)>(
                             apareto_addr[num_experts_per_node + expert])
                       : quant_type.fc1.weight_block_scale,
          num_tokens_before_expert, apareto_host ? int64_t(0) : weight_index);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''          quant_type.fc2.weight_block_scale, num_tokens_before_expert, weight_index);''',
     '''          apareto_host ? reinterpret_cast<decltype(quant_type.fc2.weight_block_scale)>(
                             apareto_addr[3 * num_experts_per_node + expert])
                       : quant_type.fc2.weight_block_scale,
          num_tokens_before_expert, apareto_host ? int64_t(0) : weight_index);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''expert, gemm1_in, weights1,''',
     '''expert, gemm1_in,
      apareto_host ? reinterpret_cast<WeightType const*>(apareto_addr[expert]) : weights1,''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''      bias1, gemm1_output, nullptr, nullptr, expert, weight_index);''',
     '''      bias1, gemm1_output, nullptr, nullptr, expert, apareto_host ? int64_t(0) : weight_index);''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''expert, gemm2_in, weights2,''',
     '''expert, gemm2_in,
      apareto_host ? reinterpret_cast<WeightType const*>(apareto_addr[2 * num_experts_per_node + expert])
                   : weights2,''', 1),
    (f'{BACKEND}/cutlass_fused_moe_kernels.cuh',
     '''      bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, expert, weight_index);''',
     '''      bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, expert,
      apareto_host ? int64_t(0) : weight_index);''', 1),
    (f'{BACKEND}/flashinfer_cutlass_fused_moe_binding.cu',
     '''    if (isNvfp4Quant() && quant_scales.has_value() && quant_scales.value().size() == 7) {''',
     '''    if (isNvfp4Quant() && quant_scales.has_value() && quant_scales.value().size() >= 7) {''', 1),
    (f'{BACKEND}/flashinfer_cutlass_fused_moe_binding.cu',
     '''      TVM_FFI_ICHECK(quant_scales.value().size() == 6 || quant_scales.value().size() == 7)
          << "Expecting 6 quant scales (7 with an apareto slot table) for nvfp4 quantization";
      bool const apareto_slots = quant_scales.value().size() == 7;''',
     '''      TVM_FFI_ICHECK(quant_scales.value().size() >= 6 && quant_scales.value().size() <= 8)
          << "Expecting 6 quant scales (7 with an apareto slot table, 8 with its host-address table) for nvfp4";
      bool const apareto_slots = quant_scales.value().size() >= 7;
      bool const apareto_host = quant_scales.value().size() == 8;''', 1),
]
# Inside the NVFP4 branch only: in slot mode the weight block scales hold the whole bank.
NVFP4_BRANCH = ('    else if (isNvfp4Quant()) {', '    } else if (mUseDeepSeekFP8BlockScaling) {')
NVFP4_EDITS = [
    ('fc1_weight_block.size(0) == num_experts_on_rank &&',
     '(apareto_slots || fc1_weight_block.size(0) == num_experts_on_rank) &&', 2),
    ('fc2_weight_block.size(0) == num_experts_on_rank &&',
     '(apareto_slots || fc2_weight_block.size(0) == num_experts_on_rank) &&', 1),
]
# Copied unchanged so the overlaid binding's relative include "../../tvm_ffi_utils.h" resolves.
UNCHANGED = ['tvm_ffi_utils.h']


def _patched_sources(csrc):
    texts = {}
    for rel, old, new, count in EDITS:
        text = texts.get(rel) or (csrc / rel).read_text()
        found = text.count(old)
        if found != count:
            raise RuntimeError(f'{rel}: expected {count} occurrence(s) of an edit anchor, found {found}')
        texts[rel] = text.replace(old, new)
    rel = f'{BACKEND}/flashinfer_cutlass_fused_moe_binding.cu'
    text = texts[rel]
    start = text.index(NVFP4_BRANCH[0])
    end = text.index(NVFP4_BRANCH[1], start)
    branch = text[start:end]
    for old, new, count in NVFP4_EDITS:
        if branch.count(old) != count:
            raise RuntimeError(f'NVFP4 branch: expected {count} of {old!r}, found {branch.count(old)}')
        branch = branch.replace(old, new)
    old_return = '      return kernels::QuantParams::FP4('
    if branch.count(old_return) != 1:
        raise RuntimeError('NVFP4 branch: expected one QuantParams::FP4 return')
    head, tail = branch.split(old_return)
    close = tail.index(');') + 2
    branch = (head + '      auto apareto_params = kernels::QuantParams::FP4(' + tail[:close] +
              '\n      if (apareto_slots) {\n        apareto_params.fp4.expert_slot =\n'
              '            static_cast<int const*>(quant_scales.value()[6].data_ptr());\n      }\n'
              '      if (apareto_host) {\n'
              '        auto const& host = quant_scales.value()[7];\n'
              '        CHECK_INPUT_TYPE(host, dl_int64);\n'
              '        CHECK_DIM(2, host);\n'
              '        TVM_FFI_ICHECK(host.size(0) == 4 && host.size(1) == num_experts_on_rank)\n'
              '            << "apareto host-address table must be [4, num_experts]";\n'
              '        apareto_params.fp4.expert_host = static_cast<int64_t const*>(host.data_ptr());\n'
              '      }\n'
              '      return apareto_params;' + tail[close:])
    texts[rel] = text[:start] + branch + text[end:]
    for rel in UNCHANGED:
        texts[rel] = (csrc / rel).read_text()
    return texts


def build_spec():
    """JitSpec for the patched module: the stock SM120 spec with three files overlaid."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
    csrc = Path(jit_env.FLASHINFER_CSRC_DIR)
    texts = _patched_sources(csrc)
    digest = hashlib.sha256(''.join(texts[k] for k in sorted(texts)).encode()).hexdigest()[:12]
    overlay = Path(jit_env.FLASHINFER_GEN_SRC_DIR) / f'apareto_moe_slots_{digest}'
    for rel, text in texts.items():
        # Same relative layout as FlashInfer's csrc, so relative includes resolve inside the overlay.
        path = overlay / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    stock = gen_cutlass_fused_moe_sm120_module()
    texts_needed = ('flashinfer_cutlass_fused_moe_binding.cu', 'cutlass_fused_moe_instantiation.cu')
    for name in texts_needed:
        target = overlay / BACKEND / name
        if not target.exists():
            target.write_text((csrc / BACKEND / name).read_text())
    replaced = {csrc / BACKEND / name: overlay / BACKEND / name for name in texts_needed}
    for source in replaced:
        if source not in stock.sources:
            raise RuntimeError(f'Stock MoE spec no longer compiles {source}')
    # The overlaid .cu files include the patched .cuh from their own directory; the patched
    # moe_kernels.h must win the -I search for every file of the module, so its directory goes first.
    return replace(stock, name=f'{MODULE}_{digest}',
                   sources=[replaced.get(source, source) for source in stock.sources],
                   extra_include_dirs=[(overlay / HEADER).parent, *(stock.extra_include_dirs or [])])


def load():
    """FlashInfer's Python MoE runner bound to the patched module (same API as flashinfer's cutlass_fused_moe)."""
    import flashinfer.fused_moe.core as core
    import shutil
    import sys
    if shutil.which('ninja') is None:   # the venv's ninja, when the venv is not on PATH (the server's case)
        os.environ['PATH'] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    spec = build_spec()
    original = core.gen_cutlass_fused_moe_sm120_module
    core.gen_cutlass_fused_moe_sm120_module = lambda *args, **kwargs: spec
    try:
        return core.get_cutlass_fused_moe_module.__wrapped__('120')
    finally:
        core.gen_cutlass_fused_moe_sm120_module = original
