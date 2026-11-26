# How it works

## 1. The problem

Qwen3.6-35B-A3B has 40 MoE layers, each with 256 routed experts: 10,240 (layer, expert) **seats**. In the NVFP4
checkpoint each expert is 1,769,496 bytes, so the routed experts alone are **16.875 GiB**. The rest of the
model (attention, gated-DeltaNet, shared experts, embeddings) adds about 4 GiB. A conventional server keeps
every expert resident, because any token can route to any of them. On a 16 GB card the model does not load.

## 2. Phase-dependent residency

Prefill and decode use the experts differently:

| | prefill | decode |
|---|---|---|
| tokens per step | a whole chunk of the prompt | one per request |
| experts touched | effectively all of them | the ones this request keeps routing to |
| each expert is used | once per chunk | on every step, for thousands of steps |

So each phase gets its own residency:

```
 request:  PREFILL ──▶ SELECT ──▶ DECODE (think) ── every 1,056 tokens: REFRESH ──▶ </think> ─▶ SWAP ─▶ DECODE (answer)
 experts:  dense,      static +   the think          probe, re-read, blend,                  R* on    the answer
           all of      midrun     keep-set           cut; load only what                     the      keep-set
           them        R*                            the new cut adds                        answer
```

1. **Prefill is dense.** Every layer runs with all 256 experts. The ones that are not in the GPU's expert bank
   are copied in from a pinned host-RAM tier as the layer runs. Quality during prefill is the stock model's.
2. **Select.** At the end of prefill, activation taps at 16 sites are read over 8 spacer rows (§3). Rankfield
   (R\*) turns them into two rankings of all 10,240 seats: the **static** think ranking, kept for the whole
   request, and a **midrun** ranking. The think keep-set is the budget's cut of their blend,
   `(1 − w) · static + w · midrun` (w = 0.5). A budget of **5,018 seats (49%)**, cut per layer, gives the keep-set.
3. **Decode on the keep-set.** Each step the router picks its top-8 experts as usual. Picks outside the keep-set
   are dropped, except for the **rescue** (§4): per token and layer at most one dropped pick, chosen by a fixed
   rule, is computed anyway.
4. **Refresh every 1,056 thinking tokens.** The spacer is run once more as a probe under the live keep-set and
   rolled back. The midrun bundle reads it, its ranking is blended with the same static ranking, and the cut
   becomes the new keep-set. Only the experts it adds are loaded (typically a few dozen).
5. **Think, then answer.** When `</think>` is sampled, the spacer runs once more as a probe after `</think>`; the
   answer bundle reads it and selects the *answer* keep-set, loaded in one swap.

Prefill, the probes and the swaps are the only places experts are loaded; decode itself never loads an expert.

## 3. Rankfield (R\*)

Rankfield (R\*) is a two-tower retrieval network (dual encoder) with an aggregation head that predicts a ranking
of the model's experts for a request.

- **Input:** the features captured at a read point: 16 activation taps × 8 rows, fp16 (180,224 values).
- **Output:** a ranking of the 10,240 seats. A per-layer quota cut gives the keep-set for a budget.
- **Bundles:** one per read point, in `artifacts/code/`: `think` (the static ranking, read after prefill),
  `midrun` (read after prefill and at every refresh), `answer` (read after `</think>`). Each is checked against
  its `MANIFEST.sha256` at boot.
- **Contract:** `scripts/rstar_emit.py` is the NumPy reference implementation. The in-engine selector calls it,
  and the blend `(1 − w) · static + w · midrun` is taken on its aggregated rankings before the cut.
  [SELECTION.md](SELECTION.md) walks through the computation and the code, file by file.
- **Per-layer quota:** a bundle's `layer_quota.json` lists the quota at a few budgets. At any other budget the
  reference interpolates (`interpolate_quota`): per-layer shares are interpolated linearly in the budget between
  the listed ones, with the equal split as anchor at 0 and at 10,240 seats, and rounded so the quota sums to the
  budget exactly. `tests/test_quota.py` checks every budget from 1 to 10,240.
- **Early prediction (speed only).** During prefill, layer 4's entry already has the taps hl2, hl4 and kvl3.
  The think bundle, read on those three taps alone, predicts the keep-set there, and the bank keeps the predicted
  experts while prefill evicts. That makes the load after prefill smaller. It never changes the keep-set.

**The spacer read.** The last prompt chunk is scheduled with a short fixed spacer appended (8 tokens). The taps
are read on the spacer rows. Afterwards the spacer is rolled back exactly: its attention KV is never read again,
and the gated-DeltaNet recurrent state is restored. The first token is sampled from the last prompt row, as
without a spacer. The refresh and post-think probes work the same way in the middle of decode: one extra
forward of the spacer under the live keep-set, read, rolled back.

## 4. The rescue rule

Dropping every pick outside the keep-set is the plain rule. The `code` profile adds one exception. Per decode
token and layer, take the router's top-8 picks that the keep-set dropped and score each:

- **think:** routing weight × 1 if the expert is in the request's static think cut, × `gout` (0.001) if not;
- **answer:** routing weight × `x / ax`, i.e. the plain weight rule at threshold `ax` (0.12).

The highest-scoring dropped pick is computed anyway if its score is above `x` (0.00014), at its routing weight;
at most `cap` (1) per token and layer. During think that means: a dropped expert from the static cut is almost
always restored, and an expert outside it only with routing weight above 0.14. A rescued expert is computed on the
GPU when the bank happens to hold it, otherwise by two reserved CPU cores from its host-RAM copy, inside the
decode step (`vllm027/cpu_experts.py`; a GPU fallback covers a late CPU answer). Prefill and the probes never
rescue. The rule needs every expert's host copy at a fixed address, so the host tier must hold the whole store
(17 GiB); the server refuses to boot otherwise. `tests/check_rescue_log.py` checks every logged rescue of a run
against the rule.

## 5. The think-end ladder

The `code` profile attaches a think-end ladder to every thinking request (`config/ladder.json`). It watches the
reasoning for repetition: at 40,000 and 45,000 generated tokens, a looping think gets a short nudge to finish
("I will output", "All set."), from 50,000 a ramp lowers the bar, and at 60,000 tokens `Done.\n</think>` is
forced, leaving at least 2,000 tokens for the answer under the 64,000 cap. Every intervention is written to
`<tag>_selection/ladder.jsonl`. It is part of the masked configuration whose LiveCodeBench result is reported
([RESULTS.md](RESULTS.md)), and a deviation from Red Hat's recipe ([RUNNING.md](RUNNING.md)). `--ladder off`
turns it off.

## 6. Memory tiers and concurrency

| tier | holds | `code` profile |
|---|---|---|
| GPU expert bank | expert slots in VRAM | 5,256 slots, 8.66 GiB |
| GPU KV cache | attention KV + gated-DeltaNet state of all live requests | 1.40 GB, 65,737-token context |
| host expert tier | pinned host-RAM copy of every expert | 17 GiB (the whole store) |
| disk | the layer store (`$APARETO_HOME/stores/layer_store/experts.apkl`) | read at startup |

[MEMORY.md](MEMORY.md) gives the measured GPU and host figures item by item.

Three requests decode at once (`max_num_seqs` 3). Every token row is dispatched under its own request's
keep-set; the bank holds the union of the live keep-sets, and what does not fit (the union is larger than 5,256
slots) is read by the MoE kernel from the pinned host copy. When the requests' contexts together outgrow the KV
cache, one request is paused: its attention blocks and recurrent state move to host RAM and come back byte for
byte, never recomputed (`vllm027/concurrent_pause.py`). A request's output depends only on its own keep-sets;
concurrency changes only the speed.

## 7. The vLLM integration

vLLM 0.27.1 is installed as a stock wheel and never modified. Every component is in `apareto_loader/vllm027/`,
registered through the `vllm.general_plugins` entry point and selected by launcher flags:

| component | flag | role |
|---|---|---|
| loader | `--load-format apareto_bank` | loads dense weights from the checkpoint and experts from the stores; sizes the bank |
| model | `--hf-overrides '{"architectures": ["AparetoBankQwen3_5Moe"]}'` | slot-indexed MoE layers: a `[expert] → slot` remap per layer, taps, per-request keep-set masks, the rescue ring |
| worker | `--worker-cls …bank_worker.BankWorker` | per-phase residency: dense prefill, selection, refreshes, the `</think>` swap, decode checks |
| scheduler | `--scheduler-cls …scheduler.AsyncBankScheduler` | async scheduling with the spacer and probe lookahead; pause and resume |
| middleware | `--middleware …middleware.BankControlMiddleware` | per-request control (`vllm_xargs`): refuses thinking-off, `select=false` and other-budget requests; attaches the ladder |
| logits processor | `--logits-processors …ladder:LadderLogitsProcessor` | the think-end ladder |
| MoE kernel | `vllm027/moe_slots.py` | a separately built FlashInfer module that indexes resident slots and host copies |
| CPU experts | `vllm027/cpu_experts.py` | the rescue's CPU path (C++/CUDA sources in `cpu_experts_src/`, built on first use) |

`scripts/serve_bank.py` is the launcher (it builds the vLLM options), and `scripts/serve.py` runs it with a
profile. At boot it checks the stock-vLLM manifest, the stores against the
checkpoint and the bundles against their manifests, then serves one warm-up request before writing `ready.json`.

## 8. What is checked, and when

- **Every boot:** vLLM tree = stock manifest; store header = checkpoint; bundles = their manifests; bank + KV +
  activation reserve + margin fit the free GPU memory, to the byte; the rescue's whole-store host tier; one
  warm-up request.
- **Every selection:** the keep-set has exactly the budget's size and the quota's per-layer counts.
- **Offline gates** (`tests/`): the quota at every budget (`test_quota.py`), the bundles (`check_bundles.py`),
  the CPU-expert and rescue kernels against their references (`test_cpu_experts.py`), and a
  served run's rescues against the rule (`check_rescue_log.py`, on a run booted with `--routing-log`).
