"""Matched, resumable TP1/TP2 serving benchmarks for the per-token FP8 PR."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "vllm-bench-collection"
PY = ROOT / ".venv/bin/python"
CLIENT = BENCH / ".venv/bin/guidellm"
SOURCE = ROOT / "vllm-triton-fp8-per-token"
HEAD = "16f20b671ef0c9a9010ab635fe1e15aad1828142"
MODEL = "RedHatAI/gemma-4-12B-it-FP8-Dynamic"
REVISION = "2083adceb2a3df0465e6021e08a7196d21dbbb44"
PORT = 8010
SUITE = "rocm10-fp8-pr-tp1-tp2-20261009-v3"
RUN = ROOT / "logs" / SUITE
RESULTS = BENCH / "results/gfx1201/gemma-4-12b-fp8-dynamic" / SUITE
CONFIG_REL = Path("vllm/model_executor/layers/quantization/utils/configs")
ARMS = ("rowwise", "triton-untuned", "triton-tuned")
KERNELS = {"rowwise": "RowWiseTorchFP8ScaledMMLinearKernel"}
KERNELS.update({arm: "TritonFp8PerTokenScaledMMKernel" for arm in ARMS[1:]})
ACTIVE = []
CLIENT_ENV = {
    "GUIDELLM__MP_MAX_PENDING_BUFFER_PERCENT": "2.0",
    "GUIDELLM__MAX_WORKER_PROCESSES": "1",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def status(phase, **values):
    data = dict(phase=phase, updated=datetime.now(timezone.utc).isoformat(), **values)
    write(RUN / "status.json", data)
    print(json.dumps(data), flush=True)


def clean_env():
    env = dict(os.environ)
    env.pop("ROCPROFILER_REGISTER_LIBRARY", None)
    return env


def command(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, env=clean_env(), **kwargs)


def git(*args, repo=SOURCE):
    return command(
        ["git", "-C", repo, *args], capture_output=True, text=True
    ).stdout.strip()


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=55)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
    if proc in ACTIVE:
        ACTIVE.remove(proc)


def interrupted(signum, _frame):
    for proc in ACTIVE[:]:
        stop(proc)
    status("interrupted", signal=signum)
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


def inventory(repo):
    return {
        p.name: sha(p)
        for p in sorted((repo / CONFIG_REL).glob("*dtype=fp8_w8a8_per_token,*.json"))
    }


def prepare(repeats):
    assert git("rev-parse", "HEAD") == HEAD
    assert not git("status", "--porcelain"), "PR source has uncommitted changes"
    repos = {"tuned": RUN / "source-tuned", "untuned": RUN / "source-untuned"}
    for kind, repo in repos.items():
        if not repo.exists():
            command(["git", "-C", SOURCE, "worktree", "add", "--detach", repo, HEAD])
        assert git("rev-parse", "HEAD", repo=repo) == HEAD
        package = repo / "vllm"
        for name in ["_version.py", *[p.name for p in (SOURCE / "vllm").glob("*.so")]]:
            link = package / name
            if not link.exists():
                link.symlink_to((SOURCE / "vllm" / name).resolve())
        if kind == "untuned":
            for path in (repo / CONFIG_REL).glob("*dtype=fp8_w8a8_per_token,*.json"):
                path.unlink()
    tuned = inventory(repos["tuned"])
    assert len(tuned) == 11 and not inventory(repos["untuned"])
    files = git(
        "ls-files", "vllm", "benchmarks/kernels/benchmark_w8a8_per_token_fp8.py"
    ).splitlines()
    sources = {name: sha(repos["tuned"] / name) for name in files}
    plan = []
    for repeat in range(1, repeats + 1):
        for tp in (1, 2):
            shift = (repeat - 1) % 3 if tp == 1 else (1 - repeat) % 3
            order = ARMS[shift:] + ARMS[:shift]
            plan.extend({"tp": tp, "arm": arm, "repeat": repeat} for arm in order)
    manifest = {
        "suite": SUITE,
        "head": HEAD,
        "model": MODEL,
        "revision": REVISION,
        "repeats": repeats,
        "plan": plan,
        "source_roots": {k: str(v) for k, v in repos.items()},
        "source_sha256": sources,
        "config_sha256": tuned,
        "workload": read(
            BENCH / "benchmarks-configs/simple-agent-multiturn-prefix-cache.json"
        ),
        "client": str(CLIENT),
        "client_env": CLIENT_ENV,
        "runner_sha256": sha(Path(__file__)),
        "method": "fresh server and caches for every arm/repeat; cyclic arm order",
    }
    path = RUN / "manifest.json"
    if path.exists():
        assert read(path) == manifest, "Manifest changed; create a new suite"
    else:
        write(path, manifest)
    return manifest


def frozen(manifest):
    for kind, name in manifest["source_roots"].items():
        repo = Path(name)
        assert git("rev-parse", "HEAD", repo=repo) == HEAD
        for rel, expected in manifest["source_sha256"].items():
            if kind == "untuned" and Path(rel).name in manifest["config_sha256"]:
                assert not (repo / rel).exists()
            else:
                assert sha(repo / rel) == expected, f"Source changed: {kind}/{rel}"
    assert (
        inventory(Path(manifest["source_roots"]["tuned"])) == manifest["config_sha256"]
    )
    assert not inventory(Path(manifest["source_roots"]["untuned"]))
    assert sha(Path(__file__)) == manifest["runner_sha256"]


def gpu_state():
    code = "import json,torch,triton; print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'triton':triton.__version__,'devices':[{'name':torch.cuda.get_device_name(i),'memory':torch.cuda.mem_get_info(i)} for i in range(torch.cuda.device_count())]}))"
    return json.loads(
        command([PY, "-c", code], capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1]
    )


def wait_free(tp):
    deadline = time.monotonic() + 3600
    while True:
        state = gpu_state()
        assert len(state["devices"]) == 2
        if all(
            state["devices"][i]["memory"][0] >= state["devices"][i]["memory"][1] * 0.91
            for i in range(tp)
        ):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs are busy: {state}")
        status("waiting-for-free-gpu", tp=tp, hardware=state)
        time.sleep(20)


def get_json(url, data=None, timeout=10):
    request = Request(
        url,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_report(path):
    data = read(path)
    assert len(data["benchmarks"]) == 4
    counts = []
    seen = set()
    for benchmark in data["benchmarks"]:
        streams = int(benchmark["config"]["strategy"]["streams"])
        assert streams in (1, 2, 4, 8) and streams not in seen
        seen.add(streams)
        requests = benchmark["requests"]
        assert not requests.get("errored") and not requests.get("incomplete")
        successful = requests["successful"]
        if len(successful) != streams * 8:
            raise ValueError(
                f"Unmatched request count: streams={streams}, got {len(successful)}, expected {streams * 8}"
            )
        conversations = defaultdict(list)
        for record in successful:
            assert record["output_tokens"] == 512
            assert all(
                math.isfinite(float(record[field])) and float(record[field]) > 0
                for field in (
                    "output_tokens_per_second",
                    "time_to_first_token_ms",
                    "request_latency",
                )
            )
            conversations[record["info"]["conversation_id"]].append(
                record["info"]["turn_index"]
            )
        assert len(conversations) == streams
        assert all(sorted(turns) == list(range(8)) for turns in conversations.values())
        events = sorted(
            (timestamp, delta)
            for r in successful
            for timestamp, delta in (
                (r["request_start_time"], 1),
                (r["request_end_time"], -1),
            )
        )
        active = peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        if peak != streams:
            raise ValueError(
                f"Configured concurrency was not attained: streams={streams}, peak={peak}"
            )
        window = max(r["request_end_time"] for r in successful) - min(
            r["request_start_time"] for r in successful
        )
        counts.append(
            {
                "streams": streams,
                "successful": len(successful),
                "errored": 0,
                "incomplete": 0,
                "peak_concurrency": peak,
                "mean_concurrency": sum(r["request_latency"] for r in successful)
                / window,
            }
        )
    return sorted(counts, key=lambda x: x["streams"])


def telemetry():
    data = {}
    for gpu in sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")):
        device = gpu.parent
        values = {}
        for field in (
            "gpu_busy_percent",
            "mem_info_vram_used",
            "pp_dpm_sclk",
            "pp_dpm_mclk",
        ):
            try:
                values[field] = (device / field).read_text().strip()
            except OSError:
                pass
        for path in (device / "hwmon").glob("hwmon*/temp1_input"):
            values["temperature_millicelsius"] = path.read_text().strip()
        data[str(device.resolve())] = values
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "devices": data}


def benchmark_once(manifest, case, folder):
    tp, arm = case["tp"], case["arm"]
    kind = "untuned" if arm == "triton-untuned" else "tuned"
    source = Path(manifest["source_roots"][kind])
    hardware = wait_free(tp)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"Port {PORT} already hosts a server")
    folder.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / SUITE / folder.name / f"attempt-{time.time_ns()}"
    env = clean_env()
    env.update(
        PYTORCH_ROCM_ARCH="gfx1201",
        CUDA_VISIBLE_DEVICES="0" if tp == 1 else "0,1",
        NCCL_PROTO="Simple",
        VLLM_ROCM_USE_AITER="1",
        VLLM_ROCM_USE_AITER_LINEAR="0",
        VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="1",
        VLLM_DISABLED_KERNELS="",
        VLLM_LOGGING_LEVEL="DEBUG",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
    )
    env["PYTHONPATH"] = str(source) + ":" + env.get("PYTHONPATH", "")
    resolved = (
        subprocess.check_output(
            [str(PY), "-c", "import vllm; print(vllm.__file__)"],
            env=env,
            cwd=source,
            text=True,
        )
        .strip()
        .splitlines()[-1]
    )
    assert Path(resolved).resolve() == (source / "vllm/__init__.py").resolve()
    cmd = [
        str(PY),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--language-model-only",
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(tp),
        "--max-num-batched-tokens",
        "8192",
        "--max-model-len",
        "32768",
        "--kv-cache-dtype",
        "fp8",
        "--load-format",
        "safetensors",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
        "--enable-prefix-caching",
        "--linear-backend",
        "torch" if arm == "rowwise" else "triton",
    ]
    report = folder / "agent-multiturn-benchmarks.json"
    workload = copy.deepcopy(manifest["workload"])
    workload["spec"]["backend"]["target"] = f"http://127.0.0.1:{PORT}"
    workload["spec"]["tokenizer"] = {
        "kind": "huggingface_auto",
        "model": str(
            Path(
                "/home/ellm/.cache/huggingface/hub/models--RedHatAI--gemma-4-12B-it-FP8-Dynamic/snapshots"
            )
            / REVISION
        ),
    }
    workload["spec"]["outputs"][0]["path"] = str(report)
    config = folder / "workload.json"
    write(config, workload)
    setup = {
        "case": case,
        "head": HEAD,
        "model": MODEL,
        "revision": REVISION,
        "source_root": str(source),
        "resolved_vllm": resolved,
        "config_sha256": inventory(source),
        "command": cmd,
        "env": {
            k: env[k]
            for k in (
                "PYTORCH_ROCM_ARCH",
                "CUDA_VISIBLE_DEVICES",
                "NCCL_PROTO",
                "VLLM_ROCM_USE_AITER",
                "VLLM_ROCM_USE_AITER_LINEAR",
                "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION",
                "VLLM_DISABLED_KERNELS",
                "VLLM_LOGGING_LEVEL",
                "VLLM_CACHE_ROOT",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        },
        "client_env": CLIENT_ENV,
        "hardware": hardware,
        "workload": workload,
        "started": datetime.now(timezone.utc).isoformat(),
    }
    write(folder / "setup.json", setup)
    status("starting-server", **case, folder=str(folder))
    with (folder / "server.log").open("w") as log:
        server = subprocess.Popen(
            cmd,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ACTIVE.append(server)
        try:
            status("server-startup", **case, server_pid=server.pid, folder=str(folder))
            deadline = time.monotonic() + 3600
            last_startup_error = "API is not ready"
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {server.returncode}; see {folder / 'server.log'}"
                    )
                try:
                    if get_json(f"http://127.0.0.1:{PORT}/v1/models").get("data"):
                        break
                except (URLError, TimeoutError, ConnectionError) as exc:
                    last_startup_error = str(exc)
                time.sleep(5)
            else:
                raise TimeoutError(
                    f"Server startup exceeded one hour: {last_startup_error}"
                )
            text = (folder / "server.log").read_text()
            selected = sorted(
                set(re.findall(r"Selected (\w+) for CompressedTensorsW8A8Fp8", text))
            )
            assert selected == [KERNELS[arm]], (arm, selected)
            assert "Capturing CUDA graphs (FULL): 100%" in text
            loaded = sorted(
                set(
                    re.findall(
                        r"Using configuration from (.*?) for W8A8 per-token", text
                    )
                )
            )
            if arm == "triton-tuned":
                assert len(loaded) == 6, loaded
            else:
                assert not loaded, loaded
            write(folder / "selected-linear-kernels.json", selected)
            smoke = get_json(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": "Reply with a short greeting."}
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                },
                timeout=180,
            )
            assert smoke["choices"][0]["message"]["content"]
            write(folder / "smoke.json", smoke)
            with (
                (folder / "benchmark.log").open("w") as client_log,
                (folder / "telemetry.jsonl").open("w") as stats,
            ):
                client_env = clean_env()
                client_env.update(CLIENT_ENV)
                client = subprocess.Popen(
                    [str(CLIENT), "run", "--config", str(config)],
                    cwd=folder,
                    env=client_env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                ACTIVE.append(client)
                try:
                    started = time.monotonic()
                    status(
                        "benchmark",
                        **case,
                        server_pid=server.pid,
                        client_pid=client.pid,
                        folder=str(folder),
                    )
                    while client.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError("Server exited during benchmark")
                        if time.monotonic() - started > 14400:
                            raise TimeoutError("GuideLLM benchmark exceeded four hours")
                        stats.write(json.dumps(telemetry()) + "\n")
                        stats.flush()
                        time.sleep(10)
                    if client.returncode:
                        raise RuntimeError(f"GuideLLM exited {client.returncode}")
                finally:
                    stop(client)
            counts = validate_report(report)
            frozen(manifest)
            write(
                folder / "success.json",
                {
                    "case": case,
                    "counts": counts,
                    "selected": selected,
                    "configs_loaded": loaded,
                    "head": HEAD,
                    "report_sha256": sha(report),
                    "completed": datetime.now(timezone.utc).isoformat(),
                },
            )
        finally:
            stop(server)
    time.sleep(5)


def run_case(manifest, case):
    name = f"tp{case['tp']}-{case['arm']}-rep{case['repeat']}"
    folder = RESULTS / name
    if (folder / "success.json").exists():
        counts = validate_report(folder / "agent-multiturn-benchmarks.json")
        accepted = read(folder / "success.json")
        assert counts == accepted["counts"] and accepted["case"] == case
        assert (
            sha(folder / "agent-multiturn-benchmarks.json") == accepted["report_sha256"]
        )
        return
    for attempt in range(1, 4):
        frozen(manifest)
        if folder.exists():
            folder.rename(folder.with_name(name + f"-failed-{time.time_ns()}"))
        folder.mkdir(parents=True)
        try:
            benchmark_once(manifest, case, folder)
            status("case-complete", **case, folder=str(folder), attempt=attempt)
            return
        except ValueError as exc:
            if not any(
                reason in str(exc)
                for reason in (
                    "Unmatched request count",
                    "Configured concurrency was not attained",
                )
            ):
                raise
            write(folder / "failure.json", {"reason": str(exc), "attempt": attempt})
            status("retrying-request-count", **case, attempt=attempt, error=str(exc))
    raise RuntimeError(
        f"Unmatched request counts after three fresh-server attempts: {name}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert args.repeats >= 2
    RUN.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/rowwise-triton-e2e.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.repeats)
        frozen(manifest)
        if args.check:
            print(
                json.dumps(
                    {
                        k: manifest[k]
                        for k in ("head", "plan", "source_roots", "method")
                    },
                    indent=2,
                )
            )
            return
        write(
            RUN / "process.json",
            {
                "pid": os.getpid(),
                "suite": SUITE,
                "head": HEAD,
                "started": datetime.now(timezone.utc).isoformat(),
            },
        )
        status("started", total=len(manifest["plan"]), repeats=args.repeats)
        for case in manifest["plan"]:
            run_case(manifest, case)
        status("benchmarks-complete", total=len(manifest["plan"]), results=str(RESULTS))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        for proc in ACTIVE[:]:
            stop(proc)
