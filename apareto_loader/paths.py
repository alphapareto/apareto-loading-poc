"""Where the release keeps everything that is not code or R* bundles.

The repository holds the code and `artifacts/` (the R* bundles) only. Python environments, caches, run outputs, the
checkpoint and the expert stores live under APARETO_HOME (default ~/.apareto); scripts/local_env.sh exports the
same locations for the shell scripts. Standard library only: scripts/serve.py imports it before any venv is active.

    $APARETO_HOME/venv           the server's Python environment (stock vLLM 0.27.1)
    $APARETO_HOME/venv-lcb       the LiveCodeBench harness's environment (lighteval)
    $APARETO_HOME/cache          every cache and temporary file (HF, torch, triton, vLLM, FlashInfer, uv, tmp)
    $APARETO_HOME/results        run outputs, one new tag per run
    $APARETO_HOME/models         the checkpoint, unless APARETO_MODEL points elsewhere
    $APARETO_HOME/stores         the expert stores, unless APARETO_STORES points elsewhere
    $APARETO_HOME/vllm_stock_manifest.sha256   the installed vLLM tree's manifest, checked at every boot
"""
import os
from pathlib import Path

MODEL_NAME = 'Qwen3.6-35B-A3B-NVFP4'


def home():
    return Path(os.environ.get('APARETO_HOME') or Path.home() / '.apareto')


def venv():
    return home() / 'venv'


def lcb_venv():
    return home() / 'venv-lcb'


def cache():
    return home() / 'cache'


def results():
    return home() / 'results'


def model():
    return Path(os.environ.get('APARETO_MODEL') or home() / 'models' / MODEL_NAME)


def stores():
    return Path(os.environ.get('APARETO_STORES') or home() / 'stores')


def stock_manifest():
    return home() / 'vllm_stock_manifest.sha256'
