# Third-party notices

apareto-loading-poc is licensed under the PolyForm Strict License 1.0.0 ([LICENSE.md](LICENSE.md)). The material below
comes from other projects and keeps its own license.

## Strata and ggml (MIT License)

The AVX2 expert kernel in `apareto_loader/vllm027/cpu_experts_src/cpu_experts.cpp` (`quant_q8`, `values32`,
`hsum8`, `row_dot`) is derived from Strata (<https://github.com/Niko1221/Strata>,
`src/kernels/cpu/nvfp4_avx2.cpp`) and from ggml's NVFP4 × Q8_0 dot product (<https://github.com/ggml-org/ggml>).

Copyright (c) 2026 Niko1221 and the Strata contributors

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FlashInfer (Apache License 2.0)

`apareto_loader/vllm027/moe_slots.py` builds a separate MoE kernel module from the FlashInfer sources installed
with vLLM, after applying the edits listed in that file. To match the lines it edits, the file quotes short
excerpts of those sources. FlashInfer itself is not distributed with this repository and the installed copy is
not modified.

FlashInfer, <https://github.com/flashinfer-ai/flashinfer>
Copyright 2025-2026 NVIDIA
Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)

Licensed under the Apache License, Version 2.0: <https://www.apache.org/licenses/LICENSE-2.0>

## Software installed as dependencies

vLLM (Apache License 2.0), PyTorch, lighteval and the other packages in `requirements.lock` and
`runners/lcb/requirements.lock` are downloaded by the installer and are not part of this repository. Each is
used under its own license.

## Benchmark recipe

`runners/lcb/litellm_config.template.yaml` follows the lighteval configuration published on the
RedHatAI/Qwen3.6-35B-A3B-NVFP4 model card (<https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4>).

## Model weights

No model weights are included. The checkpoint is downloaded from Hugging Face under its own license.
