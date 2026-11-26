# apareto-loading-poc

Serve **Qwen3.6-35B-A3B** (NVFP4, 35B parameters, 256 experts per layer) on **one 16 GB GPU**, although its
expert weights alone are 16.9 GiB, without changing the model's weights.

This is a proof of concept. It shows that **it is possible to predict, from a request's prefill, which experts
will contribute to solving it**, and to decode on those experts alone. The evidence is LiveCodeBench v6, run the
way Red Hat evaluates the same checkpoint: decoding on a predicted 49% of the experts scores within seed-to-seed
spread of the dense model, on problems the predictor was trained on and on problems it has never seen
([docs/RESULTS.md](docs/RESULTS.md)).

## The idea in one paragraph

Prefill and decode use the experts differently. Prefill pushes the whole prompt through every layer once, so
it needs every expert briefly. Decode runs one token at a time for thousands of steps, and each request keeps
coming back to a fraction of the experts. The loader therefore makes expert residency **phase-dependent**:

1. **Prefill runs dense.** Experts that are not in VRAM are streamed in from pinned host RAM, layer by layer.
2. **At the end of prefill, Rankfield (R\*) ranks the experts for this request.** Rankfield (R\*) is a two-tower
   retrieval network (dual encoder) with an aggregation head that predicts a ranking of the model's experts for a
   request.
   A budget cuts its ranking into the request's **keep-set**: by default 5,018 of the 10,240 (layer, expert)
   seats, 49%.
3. **Decode runs on that keep-set.** A router pick outside it is dropped, except that a fixed rule restores at
   most one dropped pick per token and layer. While the model thinks, R\* re-reads the state every 1,056 tokens
   and updates the keep-set; at `</think>` it selects the answer's keep-set.

The model runs on stock vLLM 0.27.1 through a plugin, and vLLM itself is not patched. Clients see a normal
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
