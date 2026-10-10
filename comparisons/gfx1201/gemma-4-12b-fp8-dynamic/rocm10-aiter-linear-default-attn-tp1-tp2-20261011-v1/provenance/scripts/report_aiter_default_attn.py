"""Validate and compare the matched default-attention AITER-linear serving runs."""

import copy
import csv
import json
import os
import re
import statistics
import subprocess

import report_fp8_pr_e2e as common
import run_aiter_default_attn as bench

OUT = bench.BENCH / "comparisons/gfx1201/gemma-4-12b-fp8-dynamic" / bench.SUITE
LABELS = {
    "linear-off": "AITER linear off · Torch RowWise",
    "linear-on": "AITER linear on · tuned AITER",
}
COLORS = {"linear-off": "#6b7280", "linear-on": "#ea8a16"}
FIELDS = common.FIELDS


def normalize(setup):
    data = copy.deepcopy(setup)
    for key in ("case", "started", "hardware"):
        data.pop(key)
    env = data["env"]
    for key in (
        "VLLM_ROCM_USE_AITER_LINEAR",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        del env[key]
    del data["workload"]["spec"]["outputs"]
    return data


def collect():
    manifest = bench.read(bench.RUN / "manifest.json")
    bench.frozen(manifest)
    helper = common.module(
        bench.BENCH / "guidellm-benchmark-compare/scripts/summarize_guidellm.py"
    )
    cases, references = {}, {}
    for case in manifest["plan"]:
        tp, arm, repeat = case["tp"], case["arm"], case["repeat"]
        folder = bench.RESULTS / f"tp{tp}-{arm}-rep{repeat}"
        report = folder / "agent-multiturn-benchmarks.json"
        success = bench.read(folder / "success.json")
        assert success["case"] == case and success["head"] == manifest["head"]
        assert success["report_sha256"] == bench.sha(report)
        assert success["counts"] == bench.validate_report(report)
        assert success["selected"] == [bench.KERNELS[arm]]
        setup = bench.read(folder / "setup.json")
        normalized = normalize(setup)
        references.setdefault(tp, normalized)
        assert normalized == references[tp], folder
        assert setup["env"]["VLLM_ROCM_USE_AITER_LINEAR"] == (
            "1" if arm == "linear-on" else "0"
        )
        assert not any(
            key.startswith("VLLM_")
            and any(s in key for s in ("ATTENTION", "ATTN", "MHA"))
            for key in setup["env"]
        )
        assert not any(
            flag in setup["command"]
            for flag in (
                "--revision",
                "--tokenizer-revision",
                "--language-model-only",
                "--enable-prefix-caching",
                "--attention-backend",
            )
        )
        assert setup["revision"] == manifest["revision"]
        assert setup["client_env"] == manifest["client_env"]
        assert setup["hardware"]["torch"] == "2.13.0+rocm10.0.0"
        assert setup["hardware"]["triton"] == "3.8.0"
        log = (folder / "server.log").read_text()
        assert bench.attention_backends(log) == success["attention"] == ["TRITON_ATTN"]
        assert "Enabling prefix caching by default" in log
        captured = sorted(
            set(
                map(
                    int,
                    re.findall(
                        r"CG Capture: mode=FULL, batch_desc=.*?num_tokens=(\d+)", log
                    ),
                )
            )
        )
        assert all(m in captured for m in (1, 2, 4, 8)), (folder, captured)
        raw = bench.read(report)
        cases[tp, arm, repeat] = {
            "report": report,
            "raw": raw,
            "means": helper.load_means(report),
            "success": success,
            "full_graph_capture_batch_sizes": captured,
            "jit_kernel_names": sorted(
                set(
                    re.findall(
                        r"Triton kernel JIT compilation during inference: (\w+)", log
                    )
                )
            ),
            "prompt_tokens": {
                str(p["config"]["strategy"]["streams"]): {
                    "mean": statistics.mean(
                        r["prompt_tokens"] for r in p["requests"]["successful"]
                    ),
                    "minimum": min(
                        r["prompt_tokens"] for r in p["requests"]["successful"]
                    ),
                    "maximum": max(
                        r["prompt_tokens"] for r in p["requests"]["successful"]
                    ),
                }
                for p in raw["benchmarks"]
            },
        }
    assert len(cases) == 4 * manifest["repeats"]
    return manifest, cases, helper


def overview(summary, field, filename, ylabel, repeats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tp, ax in zip(("1", "2"), axes):
        for i, arm in enumerate(bench.ARMS):
            values = [summary[tp][arm][str(s)][field] for s in (1, 2, 4, 8)]
            bars = ax.bar(
                np.arange(4) + (i - 0.5) * 0.32,
                [v["mean"] for v in values],
                0.32,
                yerr=[
                    [v["mean"] - v["minimum"] for v in values],
                    [v["maximum"] - v["mean"] for v in values],
                ],
                capsize=3,
                color=COLORS[arm],
                label=LABELS[arm],
            )
            ax.bar_label(
                bars,
                fmt="%.0f" if field == "time_to_first_token_ms" else "%.2f",
                padding=4,
                fontsize=9,
            )
        ax.set_xticks(np.arange(4), ["1", "2", "4", "8"])
        ax.set_xlabel("Concurrent streams")
        ax.set_ylabel(ylabel)
        ax.set_title(f"TP{tp}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Gemma 4 12B FP8 · default attention (TRITON_ATTN) · {repeats} fresh-server repetitions"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def main():
    manifest, cases, _helper = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    common.bench, common.OUT = bench, OUT
    paths, metrics = common.pool(manifest, cases)
    ratios, evidence = {}, {}
    for tp in ("1", "2"):
        ratios[tp] = {}
        for streams in ("1", "2", "4", "8"):
            ratios[tp][streams] = {}
            for field in FIELDS:
                a, b = (
                    metrics[tp]["linear-off"][streams][field],
                    metrics[tp]["linear-on"][streams][field],
                )
                ratios[tp][streams][field] = {
                    "ratio": b["mean"] / a["mean"],
                    "percent_change": 100 * (b["mean"] / a["mean"] - 1),
                    "paired_run_ratios": [
                        y / x for x, y in zip(a["run_means"], b["run_means"])
                    ],
                }
        ratios[tp]["geomeans"] = {
            field: common.geomean(
                ratios[tp][s][field]["ratio"] for s in ("1", "2", "4", "8")
            )
            for field in FIELDS
        }
        ratios[tp]["paired_tps_geomeans"] = [
            common.geomean(
                ratios[tp][s][FIELDS[0]]["paired_run_ratios"][i]
                for s in ("1", "2", "4", "8")
            )
            for i in range(manifest["repeats"])
        ]
    for (tp, arm, repeat), c in cases.items():
        evidence[f"tp{tp}-{arm}-rep{repeat}"] = {
            "path": str(c["report"]),
            "sha256": bench.sha(c["report"]),
            "counts": c["success"]["counts"],
            "selected": c["success"]["selected"],
            "prompt_tokens": c["prompt_tokens"],
            "graph_capture_sizes": c["full_graph_capture_batch_sizes"],
            "jit_kernel_names": c["jit_kernel_names"],
        }
    bench.write(
        OUT / "summary.json",
        {
            "manifest": str(bench.RUN / "manifest.json"),
            "head": manifest["head"],
            "repeats": manifest["repeats"],
            "metrics": metrics,
            "comparisons": ratios,
            "sources": evidence,
        },
    )
    with (OUT / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["tp", "streams", "field", "off_mean", "on_mean", "on_vs_off_percent"]
        )
        for tp in ("1", "2"):
            for streams in ("1", "2", "4", "8"):
                for field in FIELDS:
                    writer.writerow(
                        [
                            tp,
                            streams,
                            field,
                            metrics[tp]["linear-off"][streams][field]["mean"],
                            metrics[tp]["linear-on"][streams][field]["mean"],
                            ratios[tp][streams][field]["percent_change"],
                        ]
                    )
    figures = ("serving_tps.png", "serving_ttft.png", "serving_e2e.png")
    for field, name, label in zip(
        FIELDS,
        figures,
        (
            "Mean output tokens/s per request ↑",
            "Mean TTFT (ms) ↓",
            "Mean E2E latency (s) ↓",
        ),
    ):
        overview(metrics, field, name, label, manifest["repeats"])
    prompt_delta = max(
        abs(
            cases[tp, "linear-on", rep]["prompt_tokens"][s]["mean"]
            / cases[tp, "linear-off", rep]["prompt_tokens"][s]["mean"]
            - 1
        )
        * 100
        for tp in (1, 2)
        for rep in range(1, manifest["repeats"] + 1)
        for s in ("1", "2", "4", "8")
    )
    text = f"# AITER linear off/on: TP1 and TP2 with default attention (TRITON_ATTN)\n\nModel `{bench.MODEL}`, observed model/tokenizer main commit `{manifest['revision']}`. Two Radeon AI PRO R9700 gfx1201 GPUs, Torch 2.13.0+rocm10, HIP 7.15.26333, Triton 3.8. TP1 uses GPU0; TP2 uses GPUs 0/1. Source base `{manifest['head']}` plus the preserved local patch (SHA256 `{manifest['vllm_local_patch_sha256']}`). Frozen copies preserve both vLLM and installed AITER source/configs; hashes and original commands are retained in the [manifest]({os.path.relpath(bench.RUN / 'manifest.json', OUT)}).\n\n"
    text += "Both arms leave attention, prefix caching, and model/tokenizer revisions at their defaults, load the full model, and use `--linear-backend auto` with `VLLM_ROCM_USE_AITER=1`. There is no language-model-only flag or attention-specific environment override. The unpinned model main commit is checked before and after every case; the client tokenizer uses that matching cached snapshot. Only `VLLM_ROCM_USE_AITER_LINEAR` changes (0 → 1), besides fresh cache paths and output destinations. Server logs verify Gemma 4 defaults to TRITON_ATTN and enables prefix caching in every case, Torch RowWise when linear is off, and unshuffled AITER per-token FP8 when on. Tuned AITER configs come from the installed package, corresponding to the companion [ROCm/aiter#6245](https://github.com/ROCm/aiter/pull/6245).\n\n"
    text += f"Workload: eight-turn simple-agent conversations, 2048 new synthetic prompt tokens and exactly 512 generated output tokens per turn, streams 1/2/4/8, seed 20260715, one configured warmup request. {manifest['repeats']} fresh-server repetitions per TP/arm; alternating TP and arm order; unique empty compile caches. BF16, FP8 KV cache, prefix caching, max model length 32768, max batched tokens 8192, max sequences 8, GPU utilization 0.90. Client uses a single asynchronous worker with pending buffer 2.0. Every profile must reach its configured peak and at least 98% mean concurrency; every conversation contains all eight turns. All accepted profiles have 8/16/32/64 successful records, zero errors/incomplete requests.\n\n"
    text += f"Metrics use every successful request including warmup/cooldown. TPS is per request. Bars show pooled means; error bars show min/max independent run means. Higher TPS and lower TTFT/E2E improve performance. Geomeans weight the four concurrency levels equally. Full prompts grow from ~2k to ~20k tokens; response retokenization creates modest history differences (maximum matched mean prompt-length delta {prompt_delta:.3f}%). Fresh-cache first-use compilation costs remain included in these end-to-end serving metrics. Historical results using explicit unified attention and language-only loading are not mixed into this comparison.\n\n"
    example = bench.RESULTS / "tp1-linear-off-rep1"
    text += (
        "## Reproduction\n\n"
        "Use GuideLLM 0.7.3. Each raw case includes its exact server command and "
        f"environment in `setup.json`; see the [baseline setup]({os.path.relpath(example / 'setup.json', OUT)}) "
        f"and [recorded client workload]({os.path.relpath(example / 'workload.json', OUT)}). "
        "Copy the workload locally and adjust its backend target, tokenizer snapshot "
        "path, and output path for your environment, retaining the other settings. "
        "The measured client invocation was:\n\n"
        "```bash\n"
        "GUIDELLM__MAX_WORKER_PROCESSES=1 \\\n"
        "GUIDELLM__MP_MAX_PENDING_BUFFER_PERCENT=2.0 \\\n"
        "guidellm run --config workload.json\n"
        "```\n\n"
        "Run each TP/arm separately with a fresh server and empty compile caches. "
        "Change `VLLM_ROCM_USE_AITER_LINEAR` from 0 to 1 for the candidate, keeping "
        "the master AITER flag enabled in both arms. The manifest records the "
        "three-repetition order. Check the unpinned server's resolved model commit "
        "against the recorded commit and use the matching client tokenizer.\n\n"
        "## Results\n\n"
    )
    text += "| TP | TPS geomean, on/off | E2E geomean, on/off | Paired-run TPS GM range |\n|---:|---:|---:|---:|\n"
    for tp in ("1", "2"):
        g = ratios[tp]["geomeans"]
        runs = ratios[tp]["paired_tps_geomeans"]
        text += f"| {tp} | {g[FIELDS[0]]:.3f}x ({100 * (g[FIELDS[0]] - 1):+.1f}%) | {g[FIELDS[2]]:.3f}x ({100 * (g[FIELDS[2]] - 1):+.1f}%) | {min(runs):.3f}–{max(runs):.3f}x |\n"
    for tp in (1, 2):
        table = subprocess.check_output(
            [
                bench.BENCH / ".venv/bin/python",
                bench.BENCH
                / "guidellm-benchmark-compare/scripts/summarize_guidellm.py",
                paths[tp, "linear-off"],
                paths[tp, "linear-on"],
            ],
            text=True,
        )
        text += f"\n## TP{tp}\n\n{table}\n"
        pair = OUT / f"tp{tp}-linear-on-vs-off"
        subprocess.run(
            [
                bench.BENCH / ".venv/bin/python",
                bench.BENCH
                / "guidellm-benchmark-compare/scripts/plot_compare_multiturn.py",
                "--report",
                f"{LABELS['linear-off']}={paths[tp, 'linear-off']}",
                "--report",
                f"{LABELS['linear-on']}={paths[tp, 'linear-on']}",
                "--title",
                f"Gemma 4 12B FP8 · default attention (TRITON_ATTN) · TP{tp} · {manifest['repeats']} repetitions",
                "--output-dir",
                pair,
            ],
            check=True,
        )
        detail = f"# TP{tp}: AITER linear off → on\n\nSee [full setup](../report.md). All successful requests are included. Positive TPS and negative TTFT/E2E changes improve performance.\n\n{table}\n"
        for name in (
            "compare_tps_by_turn.png",
            "compare_ttft_by_turn.png",
            "compare_e2e_by_turn.png",
        ):
            detail += f"\n![{name}]({name})\n"
        (pair / "report.md").write_text(detail)
        text += f"[Per-turn TPS/TTFT/E2E charts](tp{tp}-linear-on-vs-off/report.md).\n"
    text += "\n## Achieved concurrency\n\n| TP | Streams | Mean active, off | Mean active, on | Peak in every run |\n|---:|---:|---:|---:|---:|\n"
    for tp in (1, 2):
        for streams in (1, 2, 4, 8):
            active = [
                statistics.mean(
                    next(
                        c
                        for c in cases[tp, arm, rep]["success"]["counts"]
                        if c["streams"] == streams
                    )["mean_concurrency"]
                    for rep in range(1, manifest["repeats"] + 1)
                )
                for arm in bench.ARMS
            ]
            text += f"| {tp} | {streams} | {active[0]:.3f} | {active[1]:.3f} | {streams} |\n"
    for name in figures:
        text += f"\n![{name}]({name})\n"
    text += "\nThese measurements compare serving performance for this model/workload/hardware. Numerical/model accuracy and multimodal validation remain separate work; successful synthetic-text requests do not establish accuracy equivalence. All repetitions are retained, with no performance-based run selection. Raw accepted reports and source/config hashes are linked below.\n\n"
    for key, value in sorted(evidence.items()):
        text += f"- [{key}]({os.path.relpath(value['path'], OUT)}) · SHA256 `{value['sha256']}`\n"
    report = OUT / "report.md"
    report.write_text(text)
    for md in OUT.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", md.read_text()):
            if not link.startswith("https://"):
                assert (md.parent / link).exists(), (md, link)
    assert len(list(OUT.rglob("*.png"))) == 9
    print(
        json.dumps(
            {
                "report": str(report),
                "geomeans": {tp: ratios[tp]["geomeans"] for tp in ("1", "2")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
