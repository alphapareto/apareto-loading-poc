"""Server-process helpers shared by the launchers: stock-vLLM check, GPU memory sampling, boot wait, teardown."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    with Path(path).open("x") as output:
        json.dump(data, output, indent=2)
        output.write("\n")


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def http_json(url, body=None, timeout=30):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_stock_manifest(manifest):
    root = Path(metadata.distribution("vllm").locate_file("")).resolve()
    for line in Path(manifest).read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Stock vLLM changed: {name}; rebuild the environment")


class MemorySampler:
    """NVML device samples cover startup and serving; worker RPC adds exact PyTorch allocator peaks."""
    def __init__(self, devices, path, interval=0.1, use_v2=False):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.devices = devices
        self.handles = {index: pynvml.nvmlDeviceGetHandleByIndex(index) for index in devices}
        self.interval = interval
        self.path = path
        self.use_v2 = use_v2
        self.stop_event = threading.Event()
        self.error = None
        self.peaks = {index: 0 for index in devices}
        self.initial = self.sample()
        self.count = 0
        self.thread = None

    def sample(self):
        samples = {}
        for index, handle in self.handles.items():
            info = self.nvml.nvmlDeviceGetMemoryInfo(handle, version=self.nvml.nvmlMemory_v2) if self.use_v2 else self.nvml.nvmlDeviceGetMemoryInfo(handle)
            samples[index] = {"used_bytes": info.used, "total_bytes": info.total,
                "compute_pids": [process.pid for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)]}
            if self.use_v2:
                samples[index].update({"driver_reserved_bytes": info.reserved,
                                      "addressable_bytes": info.total - info.reserved})
        return samples

    def require_idle(self):
        busy = {index: info["compute_pids"] for index, info in self.sample().items() if info["compute_pids"]}
        if busy:
            raise RuntimeError(f"GPUs are occupied; no existing processes were stopped: {busy}")

    def start(self):
        self.output = Path(self.path).open("x")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def require_within_budget(self, budgets):
        if self.error:
            raise RuntimeError(f"Device memory sampling failed: {self.error}")
        exceeded = {device: peak for device, peak in self.peaks.items() if peak > budgets[device]}
        if exceeded:
            raise RuntimeError(f"Measured device memory exceeded V: peaks={exceeded}, budgets={budgets}")

    def _run(self):
        try:
            while not self.stop_event.is_set():
                devices = self.sample()
                for index, info in devices.items():
                    self.peaks[index] = max(self.peaks[index], info["used_bytes"])
                self.output.write(json.dumps({"time_ns": time.time_ns(), "devices": devices}) + "\n")
                self.output.flush()
                self.count += 1
                self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
            self.output.close()
        self.nvml.nvmlShutdown()
        return {"source": "NVML v2 device used (driver reservation recorded separately)" if self.use_v2 else "NVML v1 occupied (includes driver reservation)", "sample_interval_seconds": self.interval,
                "samples": self.count, "initial": self.initial,
                "sampled_peak_bytes_by_device": self.peaks, "error": self.error,
                "limitation": "Sampled device peak; transients shorter than the interval may be missed. Worker RPC also records exact PyTorch allocator peaks."}


def stop_owned_process(process):
    if process is None:
        return
    # Every process passed here was started by this harness in its own session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=30)
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # The API process can exit before its worker children finish. Drain the
    # owned group before another fresh boot checks GPU availability.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)


def await_server(process, url, name, timeout, check_memory):
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        check_memory()
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness: code {process.returncode}")
        try:
            models = http_json(url + "/v1/models", timeout=2)
            if name not in [model["id"] for model in models["data"]]:
                raise RuntimeError("Port belongs to a different server")
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Server readiness timeout after {timeout}s: {last_error}")


def require_free_port(port):
    with socket.socket() as probe:
        # Match the HTTP server's bind semantics: TIME_WAIT from our previous
        # boot is reusable, while an existing listener still prevents binding.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))

