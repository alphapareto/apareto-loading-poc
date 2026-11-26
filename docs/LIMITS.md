# Limits of this proof of concept

## Scope

- **One benchmark.** This release serves and measures LiveCodeBench v6 (the `code` profile). Agentic serving
  (multi-turn tool use) is not part of it.
- **Thinking requests only.** Every request is served with thinking on, as LiveCodeBench runs. A request rendered
  with thinking off, one sent with `select=false`, one at another budget than the server's, and one too long for
  the read are refused with an error; the server keeps serving.
- **One model geometry.** Qwen3.6-35B-A3B: 40 MoE layers × 256 experts. The R\* bundles index (layer, expert)
  seats, so another MoE model needs its own bundles. Other quantizations of this model with the same geometry
  can be served, but only the RedHatAI NVFP4 checkpoint is tested.
- **One GPU family.** The patched MoE kernel and the CPU-expert server are built for SM120 (RTX 50-series) only.
  TP=1 and PP=1: one GPU per server.
- **Text only.** The server runs with `--language-model-only`; the vision tower is not loaded.
- **Linux x86-64**, with AVX2 for the CPU-served experts, and at least 3 cores per server (2 are reserved for the
  rescue).

## Serving

- **Up to three requests decode at once per server** (`max_num_seqs`); further requests queue. Throughput beyond
  that comes from more servers (one per GPU).
- **Prefix caching is off.** R\* reads features captured during prefill, and a cached prefix would skip that
  prefill.
- **No speculative decoding.** The model's MTP head is not served; the server refuses a speculative config.
- **Single-turn requests.** Each request is prefilled from its first token; a conversation's earlier turns are
  not reused.
- **Host RAM.** The rescue needs the whole expert store pinned in host RAM (17 GiB) plus room for paused
  requests' KV. The server needs 25 GiB free at boot and peaks at 25.9 GiB ([MEMORY.md](MEMORY.md)).
- **Long prompts cost more to prefill.** Each prefill chunk streams the experts missing from the bank over PCIe.

## Quality

- **Decode is restricted to the keep-set, plus the rescue.** Router picks outside the keep-set are dropped,
  except the at most one per token and layer the rescue rule restores. That is the trade the method makes, and
  it is why the results compare against the dense model under the same harness ([RESULTS.md](RESULTS.md)).
- **The ladder is part of the measured configuration.** The quality claim is for masked decode with the
  think-end ladder; without the ladder it is not measured here.
- **R\* bundles are fixed data.** They are not adapted per deployment, and this repository has no tooling to
  produce them.
- **Part of the benchmark is training data.** 70 of LiveCodeBench v6's 175 problems were part of R\*'s training
  data; the other 105 were held out, on purpose. [RESULTS.md](RESULTS.md) reports the two groups separately.
- **Other budgets are not measured.** The budget is a free parameter, and the per-layer quota is interpolated
  between the budgets the bundles list; only the default (5,018) has published numbers.

## Not in this release

- **Verification serving:** the audited worker and scheduler that check every dispatched expert against the
  keep-set are not included. `scripts/serve_bank.py` accepts them from outside the tree (`--audit`,
  `--worker-cls`, `--scheduler-cls`; [RUNNING.md](RUNNING.md#verification-runs)). Production serving keeps its own
  checks, and `--routing-log` with `tests/check_rescue_log.py` checks the rescue rule offline
  ([HOW-IT-WORKS.md §8](HOW-IT-WORKS.md#8-what-is-checked-and-when)).
