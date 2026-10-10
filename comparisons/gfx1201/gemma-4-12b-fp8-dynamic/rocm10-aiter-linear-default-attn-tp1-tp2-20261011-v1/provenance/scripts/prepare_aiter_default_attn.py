"""Freeze the user's vLLM edits and installed AITER for matched serving runs."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def prepare(bench, repeats):
    path = bench.RUN / "manifest.json"
    if path.exists():
        manifest = bench.read(path)
        assert manifest["repeats"] == repeats
        return manifest
    snapshot = bench.RUN / "source-vllm"
    bench.command(
        ["git", "-C", bench.SOURCE, "worktree", "add", "--detach", snapshot, bench.HEAD]
    )
    patch = subprocess.check_output(
        ["git", "-C", bench.SOURCE, "diff", "--binary", "HEAD"]
    )
    (bench.RUN / "vllm-local.patch").write_bytes(patch)
    if patch:
        subprocess.run(
            ["git", "-C", snapshot, "apply", "--binary", "-"], input=patch, check=True
        )
    for source in (bench.SOURCE / "vllm").glob("*.so"):
        (snapshot / "vllm" / source.name).symlink_to(source.resolve())
    version = bench.SOURCE / "vllm/_version.py"
    if version.exists():
        shutil.copy2(version, snapshot / "vllm/_version.py")
    aiter_root = bench.ROOT / "aiter"
    aiter_copy = bench.RUN / "source-aiter"
    aiter_copy.mkdir()
    shutil.copytree(
        aiter_root / "aiter",
        aiter_copy / "aiter",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("csrc", "hsa", "third_party", "asm", "include"):
        if (aiter_root / name).exists():
            (aiter_copy / name).symlink_to(aiter_root / name)
    env = bench.clean_env()
    env.update(
        PYTHONPATH=f"{snapshot}:{aiter_copy}:" + env.get("PYTHONPATH", ""),
        VLLM_ROCM_USE_AITER="1",
        VLLM_ROCM_USE_AITER_LINEAR="1",
    )
    code = """import json,aiter,torch,triton
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterPerTokenFp8ScaledMMLinearKernel
shapes={"1":[(8192,3840),(9216,3840),(3840,4096),(3840,8192),(30720,3840),(3840,15360)],"2":[(4096,3840),(5120,3840),(3840,2048),(3840,4096),(15360,3840),(3840,7680)]}
tuned={tp:{f"{n},{k}":rocm_aiter_ops.is_per_token_w8a8_gemm_tuned(n,k,current_platform.fp8_dtype()) for n,k in cases} for tp,cases in shapes.items()}
assert rocm_aiter_ops.is_rdna_linear_enabled()
assert AiterPerTokenFp8ScaledMMLinearKernel.is_supported()[0]
assert all(all(values.values()) for values in tuned.values()),tuned
print(json.dumps({"aiter_file":aiter.__file__,"torch":torch.__version__,"triton":triton.__version__,"tuned_shapes":tuned}))"""
    probe = json.loads(
        subprocess.check_output(
            [bench.PY, "-c", code], cwd=snapshot, env=env, text=True
        )
        .strip()
        .splitlines()[-1]
    )
    assert Path(probe["aiter_file"]).resolve() == aiter_copy / "aiter/__init__.py"
    assert probe["torch"] == "2.13.0+rocm10.0.0" and probe["triton"] == "3.8.0"
    plan = []
    for repeat in range(1, repeats + 1):
        for tp in (1, 2) if repeat % 2 else (2, 1):
            order = bench.ARMS if (repeat + tp) % 2 == 0 else bench.ARMS[::-1]
            plan.extend({"tp": tp, "arm": arm, "repeat": repeat} for arm in order)
    configs = aiter_copy / "aiter/ops/triton/configs/gfx1201/triton/gemm/gemm_a8w8"
    files = subprocess.check_output(
        ["git", "-C", snapshot, "ls-files", "vllm"], text=True
    ).splitlines()
    tracked = {name: bench.sha(snapshot / name) for name in files}
    binaries = {str(p): bench.sha(p) for p in (snapshot / "vllm").glob("*.so")}
    aiter_files = {
        str(p.relative_to(aiter_copy)): bench.sha(p)
        for p in (aiter_copy / "aiter").rglob("*")
        if p.is_file() and p.suffix in (".py", ".json", ".csv", ".so")
    }
    client_files = {
        str(p): bench.sha(p)
        for p in (bench.BENCH / ".venv/lib/python3.13/site-packages/guidellm").rglob(
            "*.py"
        )
    }
    assert client_files
    workload = bench.read(
        bench.BENCH / "benchmarks-configs/simple-agent-multiturn-prefix-cache.json"
    )
    assert workload["spec"]["profile"]["streams"] == [1, 2, 4, 8]
    assert workload["spec"]["data"] == [
        {
            "kind": "synthetic_text",
            "turns": 8,
            "prompt_tokens": 2048,
            "output_tokens": 512,
        }
    ]
    manifest = {
        "suite": bench.SUITE,
        "head": bench.HEAD,
        "model": bench.MODEL,
        "revision": bench.model_revision(),
        "revision_selection": "Unpinned model/tokenizer main; verify main commit before and after each case",
        "repeats": repeats,
        "plan": plan,
        "source_roots": {"tuned": str(snapshot)},
        "source_sha256": tracked,
        "binary_sha256": binaries,
        "aiter_source": str(aiter_copy),
        "aiter_sha256": aiter_files,
        "aiter_probe": probe,
        "aiter_git_head": bench.git("rev-parse", "HEAD", repo=aiter_root),
        "aiter_config_sha256": {p.name: bench.sha(p) for p in configs.glob("*.json")},
        "vllm_local_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "client_sha256": client_files,
        "client": str(bench.CLIENT),
        "client_env": bench.CLIENT_ENV,
        "config_sha256": {},
        "workload": workload,
        "runner_sha256": bench.sha(Path(bench.__file__)),
        "prepare_sha256": bench.sha(Path(__file__)),
        "method": "matched AITER linear off/on with model-default attention and prefix caching, full model, unpinned revisions, automatic linear selection, fresh servers/caches, alternating arm/TP order",
    }
    bench.write(path, manifest)
    return manifest


def frozen(bench, manifest):
    snapshot = Path(manifest["source_roots"]["tuned"])
    assert bench.git("rev-parse", "HEAD", repo=snapshot) == manifest["head"]
    for rel, expected in manifest["source_sha256"].items():
        assert bench.sha(snapshot / rel) == expected, rel
    for path, expected in {
        **manifest["binary_sha256"],
        **manifest["client_sha256"],
    }.items():
        assert bench.sha(Path(path)) == expected, path
    for rel, expected in manifest["aiter_sha256"].items():
        assert bench.sha(Path(manifest["aiter_source"]) / rel) == expected, rel
    assert bench.sha(Path(bench.__file__)) == manifest["runner_sha256"]
    assert bench.sha(Path(__file__)) == manifest["prepare_sha256"]
