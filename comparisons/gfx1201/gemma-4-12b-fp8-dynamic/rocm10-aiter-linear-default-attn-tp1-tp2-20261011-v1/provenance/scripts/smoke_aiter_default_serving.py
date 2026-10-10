"""Smoke-test the agreed production serving flags for TP1/TP2, linear off/on."""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vllm-aiter-fp8-per-token"
PY = ROOT / ".venv/bin/python"
MODEL = "RedHatAI/gemma-4-12B-it-FP8-Dynamic"
RUN = (
    ROOT
    / "logs"
    / (
        "aiter-default-serving-smoke-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
)
PORT = 8010
ACTIVE = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def status(phase, **kwargs):
    value = {
        "phase": phase,
        "updated": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    write(RUN / "status.json", value)
    print(json.dumps(value), flush=True)


def stop():
    global ACTIVE
    if ACTIVE is not None:
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            ACTIVE.wait(timeout=55)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
            ACTIVE.wait(timeout=30)
        ACTIVE = None


def interrupted(signum, _frame):
    stop()
    status("interrupted", signal=signum)
    raise SystemExit(128 + signum)


def api(path, payload=None, timeout=180):
    request = Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def chat(messages, max_tokens=32):
    started = time.monotonic()
    response = api(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    choice = response["choices"][0]
    assert (
        isinstance(choice["message"]["content"], str)
        and choice["message"]["content"].strip()
    ), response
    assert choice["finish_reason"] in ("stop", "length"), response
    assert 0 < response["usage"]["completion_tokens"] <= max_tokens, response
    return {"elapsed_s": time.monotonic() - started, "response": response}


def gpu_state(env):
    code = "import json,torch; print(json.dumps([{'gpu':i,'name':torch.cuda.get_device_name(i),'memory':torch.cuda.mem_get_info(i)} for i in range(torch.cuda.device_count())]))"
    return json.loads(
        subprocess.check_output([PY, "-c", code], env=env, text=True)
        .strip()
        .splitlines()[-1]
    )


def test_case(tp, linear, base_env):
    global ACTIVE
    case = f"tp{tp}-linear-{'on' if linear else 'off'}"
    folder = RUN / case
    folder.mkdir()
    hardware = gpu_state(base_env)
    assert all(
        hardware[i]["memory"][0] > 0.91 * hardware[i]["memory"][1] for i in range(tp)
    ), hardware
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", PORT)) != 0, f"Port {PORT} in use"
    env = dict(base_env)
    env.update(
        CUDA_VISIBLE_DEVICES="0" if tp == 1 else "0,1",
        NCCL_PROTO="Simple",
        VLLM_ROCM_USE_AITER="1",
        VLLM_ROCM_USE_AITER_LINEAR=str(linear),
        VLLM_LOGGING_LEVEL="DEBUG",
    )
    env["PYTHONPATH"] = str(SOURCE) + ":" + env.get("PYTHONPATH", "")
    cmd = [
        str(PY),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(tp),
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-batched-tokens",
        "8192",
        "--max-model-len",
        "32768",
        "--kv-cache-dtype",
        "fp8",
        "--load-format",
        "safetensors",
        "--linear-backend",
        "auto",
    ]
    write(
        folder / "setup.json",
        {
            "command": cmd,
            "hardware_before": hardware,
            "env": {
                k: v
                for k, v in env.items()
                if k.startswith("VLLM_")
                or k in ("CUDA_VISIBLE_DEVICES", "NCCL_PROTO", "PYTHONPATH")
            },
        },
    )
    with (folder / "server.log").open("w") as log:
        ACTIVE = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status("server-startup", case=case, server_pid=ACTIVE.pid, folder=str(folder))
        startup = time.monotonic()
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if ACTIVE.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {ACTIVE.returncode}; see {folder / 'server.log'}"
                    )
                try:
                    models = api("/v1/models", timeout=5)
                    if models.get("data"):
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(5)
            else:
                raise TimeoutError("Server startup exceeded 30 minutes")
            text = (folder / "server.log").read_text()
            selected = sorted(
                set(re.findall(r"Selected (\w+) for CompressedTensorsW8A8Fp8", text))
            )
            expected = (
                "AiterPerTokenFp8ScaledMMLinearKernel"
                if linear
                else "RowWiseTorchFP8ScaledMMLinearKernel"
            )
            assert selected == [expected], selected
            attention = sorted(
                set(
                    re.findall(
                        r"Using (\w+) backend (?:out of potential backends|\(selected via)",
                        text,
                    )
                )
            )
            assert attention, "No language attention backend found in logs"
            assert (
                "Enabling prefix caching by default" in text
                or "enable_prefix_caching=True" in text
            ), "Prefix caching not confirmed"
            assert "Capturing CUDA graphs (FULL): 100%" in text, (
                "Full graph capture incomplete"
            )
            status(
                "requests",
                case=case,
                attention=attention,
                selected=selected,
                startup_s=time.monotonic() - startup,
            )
            greeting = chat(
                [{"role": "user", "content": "Reply with a short greeting."}]
            )
            write(folder / "greeting.json", greeting)
            conversations = [
                [
                    {
                        "role": "user",
                        "content": f"My favorite number is {i + 10}. Say you understand in one sentence.",
                    }
                ]
                for i in range(8)
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                first = list(pool.map(chat, conversations))
                second_messages = [
                    messages
                    + [
                        result["response"]["choices"][0]["message"],
                        {
                            "role": "user",
                            "content": "What is my favorite number? Answer with the number only.",
                        },
                    ]
                    for messages, result in zip(conversations, first)
                ]
                second = list(pool.map(chat, second_messages))
            write(
                folder / "eight-stream-two-turn.json",
                {"first": first, "second": second},
            )
            completion = chat(
                [
                    {
                        "role": "user",
                        "content": "Write a detailed explanation of how a computer executes a program, covering the CPU, memory, storage, operating system, and input/output.",
                    }
                ],
                max_tokens=512,
            )
            write(folder / "long-completion.json", completion)
            assert ACTIVE.poll() is None, "Server died during requests"
            api("/v1/models", timeout=5)
            result = {
                "case": case,
                "tp": tp,
                "linear": bool(linear),
                "attention": attention,
                "linear_kernel": selected,
                "prefix_caching": True,
                "full_graph_capture": True,
                "successful_requests": 18,
                "completion_tokens_long_request": completion["response"]["usage"][
                    "completion_tokens"
                ],
                "completed": datetime.now(timezone.utc).isoformat(),
            }
            write(folder / "success.json", result)
            status("case-passed", **result)
            return result
        finally:
            stop()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                state = gpu_state(base_env)
                if all(
                    state[i]["memory"][0] > 0.91 * state[i]["memory"][1]
                    for i in range(tp)
                ):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"GPU memory not released after stopping {case}: {state}"
                )


def main():
    RUN.mkdir(parents=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(os.environ)
    removed = [
        k
        for k in env
        if k.startswith("VLLM_") and any(s in k for s in ("ATTENTION", "ATTN", "MHA"))
    ]
    for key in [*removed, "ROCPROFILER_REGISTER_LIBRARY"]:
        env.pop(key, None)
    patch = subprocess.check_output(["git", "-C", SOURCE, "diff", "--binary", "HEAD"])
    write(
        RUN / "manifest.json",
        {
            "source": str(SOURCE),
            "head": subprocess.check_output(
                ["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True
            ).strip(),
            "local_patch_sha256": hashlib.sha256(patch).hexdigest(),
            "removed_attention_overrides": removed,
            "runner_pid": os.getpid(),
            "note": "No model/tokenizer revision, language-model-only, explicit prefix caching, or attention backend flags. DEBUG logs enable selection verification. Text-only serving smoke; multimodal inputs are not tested.",
        },
    )
    (RUN / "vllm-local.patch").write_bytes(patch)
    with (ROOT / "logs/rowwise-triton-e2e.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            results = [test_case(tp, linear, env) for tp in (1, 2) for linear in (0, 1)]
            assert all(
                result["attention"] == results[0]["attention"] for result in results
            ), results
            write(RUN / "results.json", results)
            status(
                "complete",
                cases_passed=len(results),
                successful_requests=sum(r["successful_requests"] for r in results),
                attention=results[0]["attention"],
                results=str(RUN / "results.json"),
            )
        except Exception:
            write(RUN / "failure.json", {"traceback": traceback.format_exc()})
            status("failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
