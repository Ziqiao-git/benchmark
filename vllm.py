#!/usr/bin/env python3
"""
launch_two_vllm.py
  • Spawns two independent vLLM OpenAI-compatible servers
    – Qwen-2.5-32B-Instruct on GPUs 0-3  (port 8000)
    – QwQ-32B on GPUs 4-7                (port 8001)
  • Captures logs in ./vllm_logs/
  • Writes PID files for easy shutdown
  • Waits until each server answers /v1/models before returning
"""

import os
import subprocess
import time
import requests
from pathlib import Path
from shutil import which

# ── CONFIG ────────────────────────────────────────────────────────────────
QWEN_PATH  = "/lustre/scratch/users/guangyi.liu/ziqiao/models/Qwen2.5-32B-Instruct"
QWEN_GPUS  = "0,1,2,3"
QWEN_PORT  = 8000

QWQ_PATH   = "/lustre/scratch/users/guangyi.liu/ziqiao/models/QwQ-32B"
QWQ_GPUS   = "4,5,6,7"
QWQ_PORT   = 8001

TENSOR_PARALLEL = 4          # == number of GPUs per model
GPU_UTIL        = 0.85       # 85 % VRAM cap
READY_TIMEOUT   = 300        # seconds to wait for /v1/models
LOG_DIR         = Path("./vllm_logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def start_vllm(model_path: str, port: int, gpus: str, model_name: str) -> subprocess.Popen:
    """Launch one vLLM server in the background and return the Popen object."""
    cmd = [
        "python",
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--port", str(port),
        "--gpu-memory-utilization", str(GPU_UTIL),
        "--trust-remote-code",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    log_file = LOG_DIR / f"{model_name}.log"
    print(f">>> Launching {model_name} on GPUs {gpus} (port {port}) …")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file.open("w"),
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,           # own process group for clean kill
    )

    (LOG_DIR / f"{model_name}.pid").write_text(str(proc.pid))
    return proc

def wait_until_ready(port: int, model_name: str) -> None:
    """Block until the vLLM server responds to /v1/models or timeout."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + READY_TIMEOUT
    print(f"    Waiting for {model_name} to become ready", end="", flush=True)

    while time.time() < deadline:
        try:
            requests.get(url, timeout=2)
            print(" ready ✓")
            return
        except requests.exceptions.RequestException:
            print(".", end="", flush=True)
            time.sleep(2)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if which("nvidia-smi") is None:
        print("⚠️  Warning: nvidia-smi not found in PATH; ensure GPUs are visible.")

    # 1) Qwen-2.5-32B-Instruct
    qwen_proc = start_vllm(QWEN_PATH, QWEN_PORT, QWEN_GPUS, "Qwen2.5-32B-Instruct")
    wait_until_ready(QWEN_PORT, "Qwen2.5-32B-Instruct")

    # 2) QwQ-32B
    qwq_proc  = start_vllm(QWQ_PATH, QWQ_PORT, QWQ_GPUS, "QwQ-32B")
    wait_until_ready(QWQ_PORT, "QwQ-32B")

    print("\nAll vLLM servers are up!")
    print(f"  • Qwen2.5-32B-Instruct → http://127.0.0.1:{QWEN_PORT}/v1")
    print(f"  • QwQ-32B              → http://127.0.0.1:{QWQ_PORT}/v1")
    print("Logs: ./vllm_logs/*.log   PIDs: ./vllm_logs/*.pid")

    # Keep parent alive so children stay attached (Ctrl-C will kill them)
    try:
        qwen_proc.wait()
        qwq_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers …")
        for proc in (qwen_proc, qwq_proc):
            try:
                os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM group
            except ProcessLookupError:
                pass

if __name__ == "__main__":
    main()
