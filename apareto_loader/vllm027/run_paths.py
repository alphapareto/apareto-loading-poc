"""Per-run output paths reach the engine through the environment, not --additional-config.

vLLM hashes --additional-config into its torch.compile cache key. Per-run paths there (patch log, prefill log,
selection directory, audit path) gave every boot a new key and a fresh ~40 s compile. Passing them through
APARETO_RUN_PATHS keeps the key equal to that of any other boot with the same configuration and code.
"""
import json
import os

ENV = 'APARETO_RUN_PATHS'


def run_env(patch_log, selection_directory, prefill_log, audit_path=None):
    """Environment entries for one server: the per-run paths as JSON."""
    paths = {'apareto_patch_log': patch_log, 'apareto_selection_dir': selection_directory,
             'apareto_prefill_log': prefill_log, 'apareto_audit_path': audit_path}
    return {ENV: json.dumps({key: str(value) for key, value in paths.items() if value is not None}, sort_keys=True)}


def run_path(vllm_config, key, required=True):
    """A per-run path from APARETO_RUN_PATHS, or from --additional-config (the pool engine's launchers)."""
    paths = json.loads(os.environ.get(ENV, '{}'))
    if key in paths:
        return paths[key]
    if key in vllm_config.additional_config:
        return vllm_config.additional_config[key]
    if vllm_config.additional_config.get('apareto_run_paths_env') and required:
        raise ValueError(f'{key} missing: this launch passes per-run paths in {ENV}, which is not set')
    if required:
        raise ValueError(f'{key} is not set ({ENV} or --additional-config)')
    return None
