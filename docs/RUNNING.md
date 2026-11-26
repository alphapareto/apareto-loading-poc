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

## LiveCodeBench v6

This is Red Hat's recipe from the model card: lighteval at commit `6f0f351`, the `lcb:codegeneration_v6|0`
task (175 problems), thinking on, `max_new_tokens` 64,000, temperature 1.0, top-p 0.95, top-k 20, presence
penalty 1.5. The number is the mean of seeds 42, 1234 and 4158.

```bash
bash runners/lcb/install.sh                       # once: lighteval in ~/.apareto/venv-lcb
bash runners/lcb/run.sh 42 0                      # one seed on GPU 0
DEVICES="0 1" bash runners/lcb/run_all.sh         # all three seeds over two GPUs
python3 runners/lcb/summarize.py ~/.apareto/results/lcb/lcb_s*
```

Each `run.sh` boots a fresh server with the `code` profile, runs lighteval against it and stops the server, so
each seed is a separate boot. `SERVER_ARGS='--budget 4608'` passes options to the server. With several GPUs,
`run_all.sh` splits the machine's cores evenly between the servers. Outputs go to `$APARETO_HOME/results/lcb/<tag>/`: the
rendered `litellm_config.yaml`, lighteval's `results_*.json` and per-sample details, and the logs.

These settings differ from Red Hat's run:

| | Red Hat | here | why |
|---|---|---|---|
| `concurrent_requests` | 64 | the server's `max_num_seqs` (3) | requests beyond what the server decodes at once would queue into the 2,400 s timeout |
| `--max-model-len` | 96,000 | 65,737 | the longest LiveCodeBench v6 prompt (1,737 tokens) plus the 64,000 generated: every request gets the full 64,000 |
| think-end ladder | none | `config/ladder.json` | part of the masked configuration; at most a nudge or two and a forced `</think>` at 60,000 tokens ([HOW-IT-WORKS.md §5](HOW-IT-WORKS.md#5-the-think-end-ladder)) |

The prompts, sampling, scoring and problem set are lighteval's. `-- --ladder off` runs without the ladder.

One seed takes 14 to 16 hours on one GPU (measured, three seeds, two servers sharing one 8-core host). Many of
the 175 problems think for tens of thousands of tokens at about 45 tok/s per request, three at a time.
