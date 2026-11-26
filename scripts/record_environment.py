"""Record exact installed dependencies and the untouched vLLM file manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--vllm-path-record", required=True)
    parser.add_argument("--expected-vllm", default="0.27.1")
    args = parser.parse_args()
    for name in (args.out, args.manifest, args.vllm_path_record):
        if Path(name).exists():
            parser.error(f"{name} exists; use a new output path")
    version = metadata.version("vllm")
    if version != args.expected_vllm:
        parser.error(f"vLLM version {version} != requested {args.expected_vllm}")
    import torch
    root = Path(metadata.distribution("vllm").locate_file("vllm")).resolve()
    packages = dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions()))
    lock = Path(args.lock)
    if not lock.exists():
        with lock.open("x") as output:
            output.write("# Exact installed versions from the stock bootstrap; rebuild with this file.\n")
            output.writelines(f"{name}=={value}\n" for name, value in packages.items())
    count = 0
    with Path(args.manifest).open("x") as output:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            output.write(f"{digest}  {path.relative_to(root.parent)}\n")
            count += 1
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "total_memory_bytes": props.total_memory,
                        "compute_capability": [props.major, props.minor]})
    driver = subprocess.run(["nvidia-smi", "--query-gpu=index,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False)
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "packages": packages, "vllm": version, "vllm_path": str(root),
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "devices": devices, "driver_query": driver.stdout.strip(),
        "stock_manifest": args.manifest, "stock_file_count": count,
        "requirements_lock": str(lock),
        "requirements_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "server_boots": 0,
    }
    with Path(args.out).open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    with Path(args.vllm_path_record).open("x") as output:
        output.write(str(root) + "\n")
    print(f"[PASS] stock vLLM {version}, Python {record['python']}, torch {packages['torch']}")
    print(f"[PASS] recorded {count} stock files and {len(packages)} dependency versions")
    print(f"Environment: {args.out}")


if __name__ == "__main__":
    main()
