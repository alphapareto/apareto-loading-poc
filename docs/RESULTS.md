# Results

Every figure is labeled **measured** (with its number of boots), **predicted** (from a formula) or **assumed**
(not yet checked).

This release is a proof of concept for one claim, at the `code` profile's defaults:

> **From a request's prefill it is possible to predict which experts will contribute to solving it.** Decoding on
> the predicted 49% of the experts, with the think-end ladder, scores within seed-to-seed spread of the dense
> model on LiveCodeBench v6, both on problems the predictor was trained on and on held-out problems.

The masked side runs with the ladder ([HOW-IT-WORKS.md §5](HOW-IT-WORKS.md#5-the-think-end-ladder)); the dense
side is the unmasked model without it. The claim holds at the profile's defaults only; any other setting is your
own measurement.

## How the numbers are produced

- **Quality:** LiveCodeBench v6 pass@1 over its 175 problems, one fresh server boot per seed, seeds 42, 1234 and
  4158, reported per seed and as the mean.
- **Speed** comes from the server's per-request records (`scripts/server_stats.py`): decode tokens per second
  per phase (think, answer), selection, refresh and swap latency.
- **Memory:** GPU memory is sampled by NVML every 0.1 s for the whole run; the peak is the sample maximum. The
  startup report gives the breakdown (bank, KV, weights, activation reserve).

## Configuration (`code` profile, [config/profiles.json](../config/profiles.json))

| | |
|---|---|
| keep-set | 5,018 of 10,240 seats (49%), per-layer quota |
| think keep-set | `cut(0.5 · static + 0.5 · midrun)`, refreshed every 1,056 thinking tokens |
| answer keep-set | post-think read |
| rescue | at most 1 dropped pick per token and layer, by the static-cut rule |
| think-end ladder | on |
| requests decoding at once | 3 |
| GPU expert slots | 5,256 (8.66 GiB) |
| KV cache | 1.40 GB, 65,737-token context |
| pinned host expert tier | 17 GiB (the whole store) |

## LiveCodeBench v6

Problems solved, of 175:

| | seed 42 | seed 1234 | seed 4158 | mean |
|---|---|---|---|---|
| **dense**, no ladder, development harness (measured, 1 boot per seed) | 133 | 131 | 125 | 129.7 (74.1%) |
| **masked + ladder**, development runs (measured, 1 boot per seed) | 133 | 133 | 125 | 130.3 (74.5%) |
| **masked + ladder**, this release with lighteval (measured, 1 boot per seed) | 129 | 129 | 124 | 127.3 (72.8%) |
| Red Hat's published score (stock vLLM, whole model resident) | — | — | — | 74.67% |

### Trained and held-out problems

R\* is a trained network. The 175 problems split into two groups:

- **trained** (70 problems): the problem was part of R\*'s training data (66 of them for the answer network).
- **held out** (105 problems): the problem was kept out of the training data, on purpose. For these, R\*'s
  keep-sets are predictions for prompts it has never seen.

The split was made so that the benchmark shows both cases; `runners/lcb/trained_problems.json` lists the 70. The
same runs, split:

| | seed 42 | seed 1234 | seed 4158 | mean |
|---|---|---|---|---|
| **trained (70 problems)** | | | | |
| dense | 62 | 62 | 57 | 60.3 (86.2%) |
| masked + ladder, development | 60 | 63 | 54 | 59.0 (84.3%) |
| masked + ladder, this release | 59 | 60 | 58 | 59.0 (84.3%) |
| **held out (105 problems)** | | | | |
| dense | 71 | 69 | 68 | 69.3 (66.0%) |
| masked + ladder, development | 73 | 70 | 71 | 71.3 (67.9%) |
| masked + ladder, this release | 70 | 69 | 66 | 68.3 (65.1%) |

The two groups differ in difficulty (the held-out group has 54 hard problems of 105, the trained group 26 of 70),
so compare masked with dense inside a group, not one group with the other. On the held-out problems masked decode
is 1.0 problem below dense in the release and 2.0 above it in development; on the trained problems it is 1.3
below in both.

### How to read the differences

- The development runs used the same networks and the same selection code. The bundles shipped here store the
  query-tower weights as 16-bit floats, where the development runs used 32-bit. Replaying the three development
  runs' saved reads through this release gives the same keep-set in 19,442 of 19,483 selections (99.8%): every
  static cut, and all but 35 of 17,908 refreshes, 3 of 525 first think keep-sets and 3 of 525 answer keep-sets.
  On seed 42 the 14 that differ do so by 1 seat of 5,018 in 10 cases and by 41 to 50 seats in 4. With the 32-bit
  weights every one of the 19,483 is bit-identical.
- Both development arms were scored with lighteval's LiveCodeBench scoring of the text after `</think>`. The
  release row runs lighteval end to end against the release server. Its 525 answers were scored a second time on
  an idle host, and every outcome was the same.
- Seed-to-seed spread is about 8 problems in each arm. The release is 3.0 problems per seed below the development
  runs and 2.3 below dense. A permutation test over the six masked runs puts a gap of that size at 1.1 standard
  deviations: three seeds cannot tell it from no difference, and cannot rule out a small real one.
- Over the 175 problems, the release solved a problem more often than development on 16 and less often on 24.
- The 64,000-token cap is hit equally often on both sides: on 21, 21 and 25 answers in the release, and 19, 25
  and 23 in development.

## Speed and memory (`code` profile)

This is a proof that the experts can be predicted, not an optimised server. Three things to keep in mind when
reading the speeds:

- **No MTP.** The model's multi-token-prediction head (speculative decoding) is not used; every token is one
  full forward pass.
- **A 64k context.** The server is sized for 64,000 generated tokens after the prompt (a 65,737-token context),
  and the KV cache for that is allocated in every configuration below.
- **Placement is not tuned.** Which experts sit in GPU memory and which are read from host RAM or computed on
  the CPU follows the prediction in the simplest way. MTP and better RAM/VRAM placement are where further speed
  would come from; neither is attempted here.

### Throughput

Up to, in tokens per second:

| setup | requests at once | profile setting | with the CPU computing the experts that do not fit |
|---|---|---|---|
| 16 GB, 64k context | 3 | 196 aggregate (160 over a whole request) | **215 aggregate** (180 over a whole request) |
| 16 GB, 64k context | 1 | **104** | no gain: every expert of the keep-set is already in GPU memory, so the CPU has none to compute (decode rate 113 against 109 on the same short request, 1 boot each) |
| 12 GB (simulated), 64k context | 1 | 47 | **75** |
| 8 GB (simulated), 32k context | 1 | 13 | **49** |

- "Up to" is the highest of vLLM's 10-second generation-throughput readings in a run; for one request it is
  close to the rate over the whole request.
- **Profile setting** is the configuration the LiveCodeBench score was measured with: an expert of the keep-set
  that is not in GPU memory is read from host RAM and copied in.
- **The CPU column** adds `--cpu-experts cores=auto:6,absent=cpu,admit=off`: such an expert is computed on the
  CPU and the contents of GPU memory stay fixed during decode. The CPU did 3.6% of the expert computations at
  16 GB with three requests, 27% at 12 GB and 67% at 8 GB, with no timeouts. **Only speed has been measured
  with this setting. The CPU kernel computes from 8-bit inputs, and the LiveCodeBench score has not been
  measured with the CPU doing this share of the work.**
- Moving experts to the CPU that would fit in GPU memory is slower: one request on 16 GB with the bank cut to
  4,500 slots and 6.8% of the expert computations on the CPU decoded at 101 tokens per second against 109.
- Each cell is 1 boot on short requests (1,500 to 2,000 generated tokens of one problem per request), except the
  16 GB profile cells: the three-request figure over a whole request is 158 to 165 on the three full
  LiveCodeBench runs (3 boots), and the one-request figure is 62,397 tokens in 602 s over five problems.
- **12 GB and 8 GB** are a 16 GB RTX 5070 Ti with 3.98 and 7.96 GiB held by another process: a smaller memory
  budget on this GPU's compute, not a smaller card's speed. Budgets: [MEMORY.md](MEMORY.md#a-12-gb-card-simulated).
- **The test machine limits the transfers.** Each GPU ran on 8 PCIe 3.0 lanes (two cards in one desktop board,
  an 8-core i7-9700K). Reading experts from host RAM, and prefill, go over that link; a card on 16 lanes or a
  newer PCIe generation has two to several times the bandwidth.

### Per-request detail

| | development run, seed 42 (measured, 1 boot) | this release, seeds 42 / 1234 / 4158 (measured, 3 boots) |
|---|---|---|
| think decode tok/s per request, median | 43.4 | 45.3 / 47.0 / 41.2 |
| answer decode tok/s per request, median | 70.1 | 53.5 / 50.3 / 50.1 |
| answer decode tok/s, all requests' tokens over their seconds | 49.7 | 65.6 / 64.0 / 55.0 |
| longest request, decode time | 1,866 s (64,000 tokens) | 1,796 s (64,000 tokens) / 1,673 s (64,000) / 1,902 s (52,433) |
| refresh (probe + read + load), median / p90 | 0.20 s / 0.23 s | 0.17 / 0.19 s, 0.17 / 0.19 s, 0.16 / 0.19 s |
| peak GPU memory (NVML, device total) | 15.19 GiB | 15.18 / 15.17 / 15.21 GiB |

All columns come from `scripts/server_stats.py` on each run's records, development's included. Answer phases are
short (median about 850 tokens), so a request's answer rate depends mostly on what the other two decoding requests
are doing at that moment. The per-request median and the token-weighted rate move in opposite directions between
the two harnesses. Seed 4158 shared its host with other GPU work for most of its run, which is the likely reason its
think rate is lower (assumed).

## Correctness gates on this release

| gate | result |
|---|---|
| selection replay: the development run's saved reads through this release's selector (175 requests, seed 42) | bit-identical: 175 static cuts, 175 first think keep-sets, 6,018 refreshes, 175 answer keep-sets |
| per-layer quota at every budget 1..10,240 (`tests/test_quota.py`) | pass: exact sums, listed budgets reproduced, 10,240 = every expert |
| CPU-expert and rescue kernels (`tests/test_cpu_experts.py`) | 19 of 19 pass |
| rescue rule on a served run (`tests/check_rescue_log.py`, 6 problems, `--routing-log`) | pass: 99.990% (think) and 99.968% (answer) of token-layers match the rule exactly, 0 illegal rescues |
| prompt parity: lighteval's 175 prompts, rendered by the server's chat template, vs the development runs' prompts | identical, token for token |
| parameters: `--budget 4800`, `--max-num-seqs 1`, `--cpu-rescue off`, `--midrun-every 0`, `--ladder off` (`tests/smoke_server.py`) | each boots, serves and selects at its budget |
| peak GPU memory, 6-problem smoke | 15.18 GiB (NVML device total) |
| audited serving run (6 problems, 3 decoding at once, a forced pause every 1,024 tokens, exact audits injected from outside the tree): dispatch | 92,454 of 92,454 per-step checks clean: every allowed pick ran with its own expert's weights (bank slot or pinned host record), every disallowed pick was dropped, every rescue was the rule's pick for that request |
| same run: pause and resume | 85 forced pauses = 85 expected, 85 resumes, 170 of 170 KV onloads read back byte for byte |
| same run: selection | every keep-set the server used (7 think, 83 mid-run, 2 answer) equals the selector replayed offline on that request's read |
| same run: answer swap and expert tiers | 2 of 2 swaps at `</think>`; 0 NVMe reads, 0 short admissions, 0 forced ring waits |
| full runs: pause and resume (server log totals) | 197 / 186 / 194 pauses, all resumed, no request left paused; offloads = onloads (394 / 372 / 388); 0 errors |

### After the release cleanup

The release was cut down to the served pipeline, and its environments, caches and run outputs moved out of the
tree to `$APARETO_HOME`. Removed: the earlier compact and pool engines, the root bundle and thinking-off serving,
the seat ranking (the stores are packed and warmed in expert-id order), speculative decoding, GPU prefetch, and
the verification code (now injected from outside the tree). These gates were rerun on the result:

| gate | result |
|---|---|
| selection replay (175 requests, seed 42) | bit-identical: 175 static cuts, 175 first think keep-sets, 6,018 refreshes, 175 answer keep-sets |
| quota (`tests/test_quota.py`), bundle manifests (`tests/check_bundles.py`) | pass |
| CPU-expert and rescue kernels (`tests/test_cpu_experts.py`, idle machine) | 19 of 19 pass (P1 timing p50 82.0 µs) |
| smoke (`tests/smoke_server.py`): default, and `--budget 4800`, `--max-num-seqs 1`, `--cpu-rescue off`, `--midrun-every 0`, `--ladder off` | each boots, serves, selects at its budget, refreshes on schedule |
| refusals: thinking off, `select=false`, another budget, a raw prompt without the thinking-on generation prompt | each refused with an error; the server keeps serving |
| audited serving run (6 problems, 3 at once, a forced pause every 1,024 tokens): dispatch | 83,304 of 83,304 per-step checks clean |
| same run: pause and resume; selection; swap and tiers | 76 forced pauses = 76 expected, 76 resumes, 152 of 152 KV onloads byte for byte; 84 keep-sets (7 think, 75 mid-run, 2 answer) equal the offline replay; 2 of 2 swaps, 0 NVMe reads, 0 short admissions |
| peak GPU memory (NVML device total), 6 problems, 3 at once | 15.19 GiB (2 boots) |

### After the bundles were reduced to their interface

The R\* bundles were then re-encoded to hold only what the network reads (the file and array names in
[SELECTION.md](SELECTION.md#5-the-bundles)), the midrun rank tables were stored as 8-bit integers (every value
already fit), the query-tower weights as 16-bit floats, and multi-turn KV reuse was taken out of the tree. These gates were rerun on the result:

| gate | result |
|---|---|
| selection replay, all three development seeds (525 requests), 32-bit tower weights | bit-identical: 525 static cuts, 525 first think keep-sets, 17,908 refreshes, 525 answer keep-sets |
| the same replay with the shipped 16-bit tower weights | 19,442 of 19,483 keep-sets identical; 41 differ (35 refreshes, 3 first think keep-sets, 3 answer keep-sets) |
| quota (`tests/test_quota.py`), bundle manifests (`tests/check_bundles.py`) | pass |
| smoke (`tests/smoke_server.py`, defaults, 1 boot) | boots, serves, selects at 5,018, refreshes on schedule, refuses what it should |
| audited serving run (6 problems, 3 at once, a forced pause every 1,024 tokens, 1 boot): dispatch | 86,753 of 86,753 per-step checks clean |
| same run: pause and resume; selection; swap and tiers | 80 forced pauses = 80 expected, 80 resumes, 160 of 160 KV onloads byte for byte; 88 keep-sets (7 think, 78 mid-run, 3 answer) equal the offline replay; 3 of 3 swaps, 0 NVMe reads, 0 short admissions |
| same run: memory | GPU peak 15.20 GiB; host RAM peak 25.9 GiB ([MEMORY.md](MEMORY.md)) |
| parameters (`tests/smoke_server.py`): `--budget 4800`, `--max-num-seqs 1`, `--cpu-rescue off`, `--midrun-every 0`, `--ladder off` | each boots, serves, selects at its budget, refreshes on schedule (none with `--midrun-every 0`) |

Recorded, not gated (decoding is not guaranteed to repeat token for token): with one request at a time and the
rescue off, the same 6 problems at seed 42 gave the same tokens before and after the cleanup and on both GPUs
(2 boots each). With three at once and the rescue on, two boots agree on every prefill read and first think
keep-set and then diverge, as before the cleanup: the first rescue can be served on the CPU in one boot and on the
GPU in the other, depending on which experts the bank holds.
