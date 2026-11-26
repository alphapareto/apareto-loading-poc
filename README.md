# Apareto loading PoC

**The KV cache and hidden stream state of a Mixture-of-Experts model can be used to predict which experts it will
need for the answer.**

This repository is a proof of concept of that claim. A small network reads the model's key/value cache entries
and its hidden (residual) stream at a few layers, and predicts which of the model's 10,240 experts it will use
next. It makes the prediction when the prompt has been read, renews it every 1,056 tokens while the model
thinks, and makes it once more before the answer. The model generates with only the predicted half of its
experts, and scores within seed-to-seed spread of the full model.

## Why it matters

A Mixture-of-Experts model routes every token through a few of its experts, and which ones differs from prompt
to prompt. That makes expert placement a problem as soon as the model is larger than the GPU:

1. **The experts in use should be on the GPU.** An expert that sits in GPU memory is computed at full speed. One
   that has to be fetched from host RAM, or computed on the CPU, costs far more. How fast a model generates is
   therefore decided by whether the experts it actually uses are the ones in GPU memory.
2. **Inference engines already split the work.** Engines such as Strata and Basalt divide a model between GPU
   and CPU so that it runs on hardware that cannot hold it whole. The split is only as good as the choice of
   what goes where, and today that choice is made without knowing what the request will need.
3. **Some experts do not need to be computed at all.** The prediction also says which experts a request can do
   without. Here, half of them are never loaded for the request, and a router pick that lands on one of them is
   skipped instead of computed. That saves memory and work, not only transfer time.

Prediction turns these into decisions that can be made per request, and renewed while it runs. If the experts
a request is about to use are known ahead of time, an engine can put exactly those on the GPU, leave the rest
in host memory or on the CPU, and skip the ones the request does not need. This proof of concept shows that the
prediction is good enough to act on: the model generates on the predicted experts and keeps its score.

## The evidence

LiveCodeBench v6, 175 problems, three seeds, run the way Red Hat evaluates the same checkpoint. The predictor was
trained on 70 of the problems; the other 105 were held out on purpose.

| problems solved (mean of 3 seeds) | all experts | predicted 49% of the experts |
|---|---|---|
| **held out** (105 problems the predictor has never seen) | 69.3 (66.0%) | 68.3 (65.1%) |
| trained (70 problems) | 60.3 (86.2%) | 59.0 (84.3%) |
| all 175 | 129.7 (74.1%) | 127.3 (72.8%) |

The differences are inside the seed-to-seed spread of about 8 problems. The prediction holds on prompts the
predictor has not seen as well as on those it has. [docs/RESULTS.md](docs/RESULTS.md) has every seed and how
the numbers were produced.

## What is predicted, and from what

- **Read:** 16 activations on 8 token positions: the key and value projections of 10
  attention layers (what goes into the KV cache) and the hidden stream entering 6 layers.
- **Predict:** Rankfield (R\*) is a two-tower retrieval network (dual encoder) with an aggregation head that
  predicts a ranking of the model's experts for a request. A budget cuts the ranking into the request's
  **keep-set**: 5,018 of the 10,240 (layer, expert) seats, 49%.
- **Use:** decode runs on the keep-set. A router pick outside it is dropped, except that a fixed rule restores at
  most one dropped pick per token and layer.
- **When:** first at the end of prefill. While the model thinks, the read is repeated every 1,056 tokens and
  the keep-set updated. At `</think>` a third network predicts the keep-set for the answer.

[docs/SELECTION.md](docs/SELECTION.md) follows the prediction through the code, step by step.

## What it makes possible

If the experts a request is about to use are known in advance, the rest do not have to be in GPU memory while
it decodes. This repository uses that to serve **Qwen3.6-35B-A3B** (NVFP4, 35B parameters, 256 experts per layer) on
**one 16 GB GPU**, although its expert weights alone are 16.9 GiB, without changing the model's weights:

1. **Prefill runs dense.** Experts that are not in VRAM are streamed in from pinned host RAM, layer by layer.
2. **The prediction is made**, and the experts it names are loaded. Each renewal loads only what it adds.
3. **Decode runs on the predicted experts**, three requests at a time.

It runs on stock vLLM 0.27.1 through a plugin; vLLM itself is not patched, and clients see a normal
OpenAI-compatible server. [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md) has the full picture.

## What you need

- Linux x86-64 and an NVIDIA **SM120** GPU with 16 GB (developed on an RTX 5070 Ti), with a CUDA 13 driver and
  toolkit.
- Host RAM: 25 GiB free per server, so a 32 GB machine for one server (the whole 17 GiB expert store is pinned;
  measured peak 25.9 GiB, [docs/MEMORY.md](docs/MEMORY.md)), and at least 3 CPU cores per server.
- About 85 GB of disk, NVMe preferred: 24 GB of checkpoint, 18 GB of expert store, room to build, and about 20 GB
  of Python environments and caches. All of it goes under `~/.apareto` (`APARETO_HOME`); the repository itself
  holds only the code and the R\* bundles.

## Quick start

```bash
bash setup/install.sh                      # venv + plugin + checkpoint + expert stores (add --with-runners)
python3 scripts/serve.py --profile code    # OpenAI-compatible server on 127.0.0.1:8000, model "qwen3.6-35b"
```

The R\* bundles are in `artifacts/` (410 MB); see [docs/INSTALL.md](docs/INSTALL.md#r-bundles).

## Run the benchmark

```bash
DEVICES="0 1" bash runners/lcb/run_all.sh   # LiveCodeBench v6: lighteval 6f0f351, Red Hat's config, seeds 42/1234/4158
```

Details in [docs/RUNNING.md](docs/RUNNING.md#livecodebench-v6).

## Serving profile and parameters

The `code` profile ([config/profiles.json](config/profiles.json)) sets a 65,737-token context, a 5,018-seat
keep-set, 5,256 expert slots on the GPU, three requests decoding at once, the rescue rule and the think-end
ladder. Every one of these is a startup parameter: `python3 scripts/serve.py --profile code -- --budget 4608`
serves a smaller keep-set, `--budget 10240` the unmasked model ([docs/RUNNING.md](docs/RUNNING.md#parameters)).

## Results

Quality, speed and memory are reported in [docs/RESULTS.md](docs/RESULTS.md). Each figure is labeled measured
(with its number of boots), predicted or assumed.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md): installation step by step, paths, R\* bundles, troubleshooting
- [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md): phases, selection and refresh, the rescue rule, the ladder, memory tiers, the vLLM plugin
- [docs/SELECTION.md](docs/SELECTION.md): how the keep-sets are computed, step by step, with the file and function for each step
- [docs/MEMORY.md](docs/MEMORY.md): GPU memory, host RAM and disk, item by item
- [docs/RUNNING.md](docs/RUNNING.md): the server and its parameters, the benchmark runner, and the outputs
- [docs/RESULTS.md](docs/RESULTS.md): what is measured, how, and the numbers
- [docs/LIMITS.md](docs/LIMITS.md): what this PoC does not do
- [AGENTS.md](AGENTS.md): rules for working on the code

## License

Source-available, for noncommercial use: the [PolyForm Strict License 1.0.0](LICENSE.md). You may run and study
the software for noncommercial purposes; distributing it, changing it, or building other software from it is not
permitted. The methods it implements are patent pending. Copyright (c) 2026 apareto.eu; for any other use,
contact apareto.eu.

Third-party material keeps its own license ([NOTICE.md](NOTICE.md)). The model weights are under their own
license (see the [model card](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)).
