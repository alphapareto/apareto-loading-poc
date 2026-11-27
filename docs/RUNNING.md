# Running

## The server

```bash
python3 scripts/serve.py --profile code [--device 0] [--port 8000] [-- <option> <value> ...]
```

The server is OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`). It serves the model
as `qwen3.6-35b` with the `qwen3` reasoning parser, as Red Hat serves this checkpoint for its evaluations
(`vllm serve ... --reasoning-parser qwen3 --language-model-only`). A harness that works against their vLLM
deployment should therefore work here unmodified. Tool calling is off unless `--tool-call-parser` is given (e.g.
`qwen3_coder`, with auto tool choice). Only thinking requests are served; others are refused with an error
([LIMITS.md](LIMITS.md)).

- It binds to 127.0.0.1; change it with `--host`.
- Up to three requests decode at once (`max_num_seqs`); further requests queue in the server.
- Startup takes about 6 minutes with warm caches: the dense weights, the expert bank and host tier,
  `torch.compile`, CUDA graphs, then one warm-up request. `$APARETO_HOME/results/<tag>_ready.json` appears when the server is
  ready.
- `--dry-run` prints the resolved command.
- Stop the server with SIGINT or SIGTERM. Then it writes `$APARETO_HOME/results/<tag>_results.json`.

### Parameters

Every setting of the `code` profile ([config/profiles.json](../config/profiles.json)) is a startup option of
`scripts/serve_bank.py`. Pass any of them after `--` to override the profile, e.g.
`python3 scripts/serve.py --profile code -- --budget 4608 --max-num-seqs 1`. The defaults are the configuration
the published LiveCodeBench numbers were measured with; other settings are your own measurement.

| option | default | meaning |
|---|---|---|
| `--budget` | 5,018 | seats each request keeps, of 10,240. Any value from 1 to 10,240; the per-layer quota is the bundle's at the budgets it lists and interpolated between them ([HOW-IT-WORKS.md §3](HOW-IT-WORKS.md#3-rankfield-r)). `10240` keeps every expert: the unmasked model. |
| `--gpu-experts` | 5,256 | expert slots in VRAM. Experts of the live keep-sets beyond it are read from host RAM (slower). |
| `--host-tier-gib` | 17 | pinned host RAM for expert copies; 17 holds the whole 16.9 GiB store, which the rescue needs |
| `--kv-cache-memory-bytes` | 1,400,000,000 | KV cache for all live requests |
| `--activation-reserve-bytes`, `--gpu-margin-bytes` | 250 MB, 200 MB | the rest of the exact-byte GPU fit check |
| `--max-num-seqs` | 3 | requests decoding at once; 1 is the single-request mode |
| `--max-model-len` | 65,737 | context: 64,000 generated tokens plus the longest LiveCodeBench prompt |
| `--max-num-batched-tokens` | 1,024 | prefill chunk |
| `--midrun-every` | 1,056 | thinking tokens between keep-set refreshes; `0` keeps the first think keep-set to the end |
| `--midrun-weight` | 0.5 | w in `(1 − w) · static + w · midrun` |
| `--midrun-at-prefill` | on | the first think keep-set is the blend too; `--no-midrun-at-prefill`: the static cut alone |
| `--cpu-rescue` | `x=0.00014,ax=0.12,gout=0.001,cap=1,phases=think+answer` | the rescue rule ([HOW-IT-WORKS.md §4](HOW-IT-WORKS.md#4-the-rescue-rule)); `off` drops every pick outside the keep-set |
| `--cpu-experts` | `cores=auto:2,absent=gpu` | the rescue's CPU path: the last 2 cores of the server's affinity, reserved for it |
| `--ladder` | `config/ladder.json` | the think-end ladder ([HOW-IT-WORKS.md §5](HOW-IT-WORKS.md#5-the-think-end-ladder)); `off` turns it off |
| `--think-bundle`, `--midrun-bundle`, `--answer-bundle` | `artifacts/code/…` | the R\* bundles |
| `--routing-log` | off | verification only: logs every decode step's router picks for `tests/check_rescue_log.py` |
| `--async-scheduling` | on | vLLM's asynchronous scheduling. Set by the profile (`"async_scheduling": true`); there is no command-line switch to turn it off |
| `--enforce-eager` | off | run without `torch.compile` and CUDA graphs: slower, for measurements that need module hooks |
| `--boot-seconds` | 2,400 | how long the launcher waits for the server to become ready before giving up |
| `--served-model-name` | `qwen3.6-35b` | the model name clients use |
| `--tag` | `<profile>_<UTC time>` | prefix of this run's output files; an existing tag is refused |
| `--tool-call-parser`, `--reasoning-parser` | none, `qwen3` | passed to vLLM |
| `--audit`, `--worker-cls`, `--scheduler-cls` | off | verification runs, see below |

`--model`, `--stores`, `--artifacts`, `--device`, `--port`, `--host`, `--served-model-name`, `--tag`,
`--tool-call-parser`, `--reasoning-parser` and `--dry-run` are options of `scripts/serve.py` itself and go before
the `--`. `scripts/serve.py --help` and `scripts/serve_bank.py --help` list everything.

The server refuses to boot on a combination that cannot work and names the numbers that conflict: the rescue
with a host tier smaller than the whole store, a bank or KV cache that does not fit the free GPU memory, a budget
outside 1..10,240.

Every run writes to `$APARETO_HOME/results/<tag>_*` (default `~/.apareto/results`), and the tag is new on every boot:

| file | contents |
|---|---|
| `<tag>_server.log` | the vLLM log |
| `<tag>_memory.jsonl` | NVML GPU memory samples, every 0.1 s |
| `<tag>_selection/selections.jsonl` | one row per selection, refresh and swap, and per request its decode tokens and time per phase |
| `<tag>_selection/ladder.jsonl` | every ladder event |
| `<tag>_ready.json`, `<tag>_results.json` | boot report (memory breakdown, KV capacity); final report |

`python3 scripts/server_stats.py ~/.apareto/results/<tag>` summarizes them. It reports decode tok/s per phase, selection
and swap latency, and peak and startup memory.

### Verification runs

The tree ships no audit code. `scripts/serve_bank.py` takes an audited worker and scheduler from outside it:
`--audit` gives them an audit path (`$APARETO_HOME/results/<tag>_audit.jsonl`) and the router's picks
(`--enable-return-routed-experts`), and `--worker-cls` / `--scheduler-cls` name their classes. Put their package
on the server's `PYTHONPATH` and pass the options after `--`, e.g.
`python3 scripts/serve.py --profile code -- --audit --worker-cls <package>.AuditedBankWorker --scheduler-cls
<package>.AsyncAuditedScheduler`. The audited classes subclass `BankWorker` and `AsyncBankScheduler`.

### Measurement switches

Environment variables for measuring the server. None is part of a served configuration and all are off unless
set; each that takes a path refuses an existing one.

| variable | value | effect |
|---|---|---|
| `APARETO_HOST_PHASES` | `1` | adds the host seconds of each phase between decode steps to every request's decode statistics (`phase_<name>`) |
| `APARETO_STEP_TIMING` | a new file | per step, the host time in the worker's own code, in vLLM's model step, in sampling, and between steps |
| `APARETO_PROFILE_DECODE` | `<new directory>:<context>,<context>,…` | the torch profiler records 20 decode steps at each context length and writes per-kernel CUDA time to `profile_<context>.txt` |
| `APARETO_MEMORY_SNAPSHOT` | a new directory | dumps the live GPU allocations at the end of each of the first 3 requests |
| `APARETO_CPUX_NUMERICS` | `<new file>[:steps]`, with `--enforce-eager` | compares the GPU path, the CPU kernel and a 64-bit reference for the experts computed outside GPU memory |

## Cards with 16, 12 and 8 GB

The GPU memory splits into a part that is always needed and two parts you size:

| part | size | set by |
|---|---|---|
| the model outside the experts, CUDA context, kernels, activations | about 5.2 GiB | fixed |
| KV cache | 21 KB per token of context: 1.30 GiB for the 64k context | `--max-model-len`, `--kv-cache-memory-bytes` |
| expert bank | 1.77 MB per slot | `--gpu-experts` (at least 512) |

Keep 0.2 to 0.3 GiB free. Experts of a keep-set that have no slot are read from host RAM when they are used,
which works the same way and is slower.

Each setup below is a profile in [config/profiles.json](../config/profiles.json), so it starts with one option:

| card | profile | requests at once | context |
|---|---|---|---|
| 16 GB, as benchmarked | `code` | 3 | 64k |
| 16 GB, fastest | `code-16gb-fast` | 3 | 64k |
| 12 GB | `code-12gb` | 1 | 64k |
| 8 GB | `code-8gb` | 1 | 32k |

The LiveCodeBench score was measured with `code` only. Options after `--` still override a profile's values.

### 16 GB (measured)

The `code` profile as it is: 5,256 slots, 64k context, three requests at once.

```bash
python3 scripts/serve.py --profile code
```

```bash
python3 scripts/serve.py --profile code-16gb-fast
```

The same with `--cpu-experts cores=auto:6,absent=cpu,admit=off`: up to 215 tokens per second aggregate with
three requests. With one request (`--profile code -- --max-num-seqs 1`) up to 104. Peak 15.2 GiB. The
LiveCodeBench score was measured with `code`, which reaches up to 196.

### 12 GB (measured on a simulated card, one request at a time)

```bash
python3 scripts/serve.py --profile code-12gb
```

The profile is `code` with `--max-num-seqs 1 --gpu-experts 2880 --gpu-margin-bytes 150000000
--cpu-experts cores=auto:6,absent=cpu,admit=off`. Up to 75 tokens per second, peak 11.2 GiB. At the default
margin the fit check accepts 2,850 slots. The measurement was
made on a 16 GB card with 3.98 GiB held by another process
([MEMORY.md](MEMORY.md#a-12-gb-card-simulated)), so a real 12 GB card may fit a few slots more or fewer. Three
requests at once on 12 GB have not been measured.

### 8 GB (measured on a simulated card, 32k context, one request at a time)

An 8 GB card has about 7.5 GiB addressable. After the fixed 5.2 GiB and 0.3 GiB free, about 2.0 GiB is left for
the KV cache and the bank together, so the context has to shrink:

| context | KV cache | bank |
|---|---|---|
| 64k (65,737 tokens) | 1.30 GiB | about 420 slots: below the minimum of 512, the server refuses |
| 32k generated (33,737 tokens) | 0.69 GiB | 830 slots (measured) |
| 16k generated (17,737 tokens) | 0.35 GiB | about 1,000 slots (predicted) |

```bash
python3 scripts/serve.py --profile code-8gb
```

The profile is `code` with `--max-num-seqs 1 --max-model-len 33737 --kv-cache-memory-bytes 740000000
--gpu-experts 830 --gpu-margin-bytes 150000000 --cpu-experts cores=auto:6,absent=cpu,admit=off`.

Up to 49 tokens per second, peak 7.24 GiB (measured on a 16 GB card with 7.96 GiB held by another process).
Here the `--cpu-experts` option is what makes it usable: without it about 110 experts are copied into GPU memory
for every token. Of the 175 LiveCodeBench problems, 60 stayed within a 32k context in all three published runs; the benchmark
as published needs 64k.

### The `--cpu-experts` setting

The value is a comma-separated list:

| item | values | meaning |
|---|---|---|
| `cores` | `auto:N`, or a list such as `4-7` or `6;7` | the cores the CPU expert threads run on, kept free of vLLM's own threads. `auto:N` takes the last N cores of the server's affinity and needs at least one more core for vLLM |
| `absent` | `gpu` (profile `code`), `cpu` | what happens to an expert of the keep-set that is not in GPU memory: `gpu` reads it from host RAM and computes it on the GPU; `cpu` computes it on the CPU |
| `admit` | `on` (default), `off` | `on` copies such experts into GPU memory during decode, replacing others; `off` leaves the contents of GPU memory unchanged during decode |
| `timeout_us`, `per_pair_us` | 1,000 and 150 | the GPU waits `timeout_us` plus `per_pair_us` per (token, expert) pair for the CPU's result, then computes it itself |

The `code` profile's setting, `cores=auto:2,absent=gpu`, uses the CPU for the rescue rule only. The other
profiles use `cores=auto:6,absent=cpu,admit=off`, the fastest of the variants measured.

The speeds with `absent=cpu` are measured. The LiveCodeBench score with it is not: the CPU kernel computes from
8-bit inputs, and in these setups it does from 4% (16 GB, three requests) to 67% (8 GB) of the expert
computations. Treat it as a faster setting whose quality you have to check for your use.

### Finding the bank size for a card

The exact number depends on the driver and on what else uses the GPU, so it is found by booting:

1. Start with the estimate: (addressable memory − 5.2 GiB − KV cache − 0.3 GiB) / 1.77 MB.
2. If the server refuses to start, its error states how many bytes the bank, KV cache, reserve and margin exceed
   the free memory by. Lower `--gpu-experts` by that amount divided by 1.77 MB.
3. After a run, `python3 scripts/server_stats.py ~/.apareto/results/<tag>` prints `gpu_peak_used`. If more than
   0.3 GiB is left, raise `--gpu-experts`; each 100 slots are 0.17 GiB.

The GPU must be otherwise unused: the launcher refuses a GPU that already has a process on it.

## LiveCodeBench v6

This is Red Hat's recipe from the model card: lighteval at commit `6f0f351`, the `lcb:codegeneration_v6|0`
task (175 problems), thinking on, `max_new_tokens` 64,000, temperature 1.0, top-p 0.95, top-k 20, presence
penalty 1.5. The number is the mean of seeds 42, 1234 and 4158.

```bash
bash runners/lcb/install.sh                       # once: lighteval in ~/.apareto/venv-lcb
bash runners/lcb/run.sh 42 0                      # one seed on GPU 0
DEVICES="0 1" bash runners/lcb/run_all.sh         # all three seeds over two GPUs
bash runners/lcb/run_multi.sh 42 "0 1"            # one seed spread over two GPUs
python3 runners/lcb/summarize.py ~/.apareto/results/lcb/lcb_s*
```

Each `run.sh` boots a fresh server with the `code` profile, runs lighteval against it and stops the server, so
each seed is a separate boot. `SERVER_ARGS='--budget 4608'` passes options to the server. With several GPUs,
`run_all.sh` splits the machine's cores evenly between the servers. Outputs go to `$APARETO_HOME/results/lcb/<tag>/`: the
rendered `litellm_config.yaml`, lighteval's `results_*.json` and per-sample details, and the logs.

`run_multi.sh` runs one seed on several GPUs at once: one server per GPU, and `runners/lcb/balance.py` in front
of them, which sends each request to the server with the fewest requests in flight and changes nothing in it.
lighteval runs once, with as many concurrent requests as the servers decode together. The published numbers were
measured with `run.sh`, one server per seed.

These settings differ from Red Hat's run:

| | Red Hat | here | why |
|---|---|---|---|
| `concurrent_requests` | 64 | the server's `max_num_seqs` (3) | requests beyond what the server decodes at once would queue into the 2,400 s timeout |
| `--max-model-len` | 96,000 | 65,737 | the longest LiveCodeBench v6 prompt (1,737 tokens) plus the 64,000 generated: every request gets the full 64,000 |
| think-end ladder | none | `config/ladder.json` | part of the masked configuration; at most a nudge or two and a forced `</think>` at 60,000 tokens ([HOW-IT-WORKS.md §5](HOW-IT-WORKS.md#5-the-think-end-ladder)) |

The prompts, sampling, scoring and problem set are lighteval's. `-- --ladder off` runs without the ladder.

One seed takes 14 to 16 hours on one GPU (measured, three seeds, two servers sharing one 8-core host). Many of
the 175 problems think for tens of thousands of tokens at about 45 tok/s per request, three at a time.
