"""Export the validated AITER-only suite for PR60672 without publishing it."""

import json
import os
import re
import shutil
from pathlib import Path

import report_aiter_default_attn as report
import run_aiter_default_attn as bench

TARGET = bench.ROOT / "vllm-bench-aiter-pr60672"
BRANCH = "benchmark/aiter-linear-pr60672-default-attn"
FILES = (
    "agent-multiturn-benchmarks.json",
    "benchmark.log",
    "selected-linear-kernels.json",
    "selected-attention-backends.json",
    "server.log",
    "setup.json",
    "smoke.json",
    "success.json",
    "telemetry.jsonl",
    "workload.json",
)


def main():
    manifest, cases, _ = report.collect()
    assert bench.read(bench.RUN / "status.json")["phase"] == "complete"
    assert len(cases) == 12 and manifest["repeats"] == 3
    assert len(list(report.OUT.rglob("*.png"))) == 9
    summary = bench.read(report.OUT / "summary.json")
    assert summary["head"] == manifest["head"]
    for (tp, arm, rep), case in cases.items():
        evidence = summary["sources"][f"tp{tp}-{arm}-rep{rep}"]
        assert evidence["sha256"] == bench.sha(case["report"])
        assert evidence["counts"] == case["success"]["counts"]
    if not TARGET.exists():
        bench.command(
            ["git", "-C", bench.BENCH, "worktree", "add", "-b", BRANCH, TARGET, "HEAD"]
        )
    assert bench.git("branch", "--show-current", repo=TARGET) == BRANCH
    assert not bench.git("status", "--porcelain", repo=TARGET)
    destination = TARGET / report.OUT.relative_to(bench.BENCH)
    assert not destination.exists(), (
        "Existing export must be inspected before replacement"
    )
    shutil.copytree(report.OUT, destination)
    provenance = destination / "provenance"
    provenance.mkdir()
    for name in (
        "manifest.json",
        "vllm-local.patch",
        "process.json",
        "status.json",
        "pipeline.log",
    ):
        shutil.copy2(bench.RUN / name, provenance / name)
    scripts = provenance / "scripts"
    scripts.mkdir()
    for name in (
        "run_aiter_default_attn.py",
        "prepare_aiter_default_attn.py",
        "report_aiter_default_attn.py",
        "export_aiter_default_attn.py",
        "render_aiter_default_attn_pr.py",
        "report_fp8_pr_e2e.py",
        "run_fp8_pr_e2e.py",
        "smoke_aiter_default_serving.py",
    ):
        shutil.copy2(bench.ROOT / "scripts" / name, scripts / name)
    assert bench.sha(scripts / "run_aiter_default_attn.py") == manifest["runner_sha256"]
    assert (
        bench.sha(scripts / "prepare_aiter_default_attn.py")
        == manifest["prepare_sha256"]
    )
    configs = provenance / "aiter-configs"
    configs.mkdir()
    source_configs = (
        Path(manifest["aiter_source"])
        / "aiter/ops/triton/configs/gfx1201/triton/gemm/gemm_a8w8"
    )
    for name, expected in manifest["aiter_config_sha256"].items():
        shutil.copy2(source_configs / name, configs / name)
        assert bench.sha(configs / name) == expected
    for case in cases.values():
        folder = case["report"].parent
        target = TARGET / folder.relative_to(bench.BENCH)
        target.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(folder / name, target / name)
    document = destination / "report.md"
    text = document.read_text()
    original_link = os.path.relpath(bench.RUN / "manifest.json", report.OUT)
    assert original_link in text
    document.write_text(text.replace(original_link, "provenance/manifest.json"))
    for markdown in destination.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", markdown.read_text()):
            if not link.startswith("https://"):
                assert (markdown.parent / link).exists(), (markdown, link)
    exported = [
        *destination.rglob("*"),
        *(TARGET / bench.RESULTS.relative_to(bench.BENCH)).rglob("*"),
    ]
    hashes = {
        str(path.relative_to(TARGET)): bench.sha(path)
        for path in sorted(exported)
        if path.is_file()
    }
    assert all((TARGET / name).stat().st_size < 95 * 1024 * 1024 for name in hashes)
    bench.write(destination / "artifact-sha256.json", hashes)
    bench.write(
        bench.RUN / "export.json",
        {
            "target": str(TARGET),
            "branch": BRANCH,
            "destination": str(destination),
            "files": len(hashes),
        },
    )
    print(json.dumps({"target": str(TARGET), "branch": BRANCH, "files": len(hashes)}))


if __name__ == "__main__":
    main()
