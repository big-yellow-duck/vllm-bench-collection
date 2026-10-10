"""Validate all six serving cases and plot their three independent repetitions."""

import copy
import csv
import importlib.util
import json
import math
import re
import statistics
import subprocess

import run_fp8_pr_e2e as bench

OUT = bench.BENCH / "comparisons/gfx1201/gemma-4-12b-fp8-dynamic" / bench.SUITE
LABELS = {
    "rowwise": "Torch RowWise",
    "triton-untuned": "Native Triton · heuristic",
    "triton-tuned": "Native Triton · tuned",
}
COLORS = {"rowwise": "#6b7280", "triton-untuned": "#2563eb", "triton-tuned": "#16a34a"}
FIELDS = ("output_tokens_per_second", "time_to_first_token_ms", "request_latency")


def module(path):
    spec = importlib.util.spec_from_file_location("guidellm_summary", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def normalize(setup):
    command = list(setup["command"])
    assert command[-2] == "--linear-backend"
    del command[-2:]
    env = dict(setup["env"])
    for key in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        del env[key]
    workload = copy.deepcopy(setup["workload"])
    del workload["spec"]["outputs"]
    return {
        "command": command,
        "env": env,
        "client_env": setup["client_env"],
        "workload": workload,
        "model": setup["model"],
        "revision": setup["revision"],
        "head": setup["head"],
    }


def collect():
    manifest = bench.read(bench.RUN / "manifest.json")
    bench.frozen(manifest)
    helper = module(
        bench.BENCH / "guidellm-benchmark-compare/scripts/summarize_guidellm.py"
    )
    cases = {}
    reference = {}
    for case in manifest["plan"]:
        tp, arm, repeat = case["tp"], case["arm"], case["repeat"]
        folder = bench.RESULTS / f"tp{tp}-{arm}-rep{repeat}"
        success = bench.read(folder / "success.json")
        report = folder / "agent-multiturn-benchmarks.json"
        assert bench.validate_report(report) == success["counts"]
        assert bench.sha(report) == success["report_sha256"]
        assert success["case"] == case and success["head"] == bench.HEAD
        assert success["selected"] == [bench.KERNELS[arm]]
        setup = bench.read(folder / "setup.json")
        assert setup["client_env"] == manifest["client_env"]
        normalized = normalize(setup)
        reference.setdefault(tp, normalized)
        assert normalized == reference[tp], f"Serving/workload mismatch: {folder}"
        assert setup["hardware"]["torch"] == "2.13.0+rocm10.0.0"
        assert setup["hardware"]["triton"] == "3.8.0"
        expected = {} if arm == "triton-untuned" else manifest["config_sha256"]
        assert setup["config_sha256"] == expected
        loaded = set(success["configs_loaded"])
        if arm == "triton-tuned":
            config_paths = {
                str(bench.Path(setup["source_root"]) / bench.CONFIG_REL / name)
                for name in manifest["config_sha256"]
            }
            assert len(loaded) == 6 and loaded <= config_paths
        else:
            assert not loaded
        server_log = (folder / "server.log").read_text()
        capture_batches = sorted(
            set(
                map(
                    int,
                    re.findall(
                        r"CG Capture: mode=FULL, batch_desc=.*?num_tokens=(\d+)",
                        server_log,
                    ),
                )
            )
        )
        assert capture_batches == [1, 2, 4, 8], folder
        cases[tp, arm, repeat] = {
            "folder": folder,
            "report": report,
            "raw": bench.read(report),
            "means": helper.load_means(report),
            "success": success,
            "full_graph_capture_batch_sizes": capture_batches,
            "prompt_tokens": {
                str(profile["config"]["strategy"]["streams"]): {
                    "mean": statistics.mean(
                        record["prompt_tokens"]
                        for record in profile["requests"]["successful"]
                    ),
                    "minimum": min(
                        record["prompt_tokens"]
                        for record in profile["requests"]["successful"]
                    ),
                    "maximum": max(
                        record["prompt_tokens"]
                        for record in profile["requests"]["successful"]
                    ),
                }
                for profile in bench.read(report)["benchmarks"]
            },
            "jit_kernel_names": sorted(
                set(
                    re.findall(
                        r"Triton kernel JIT compilation during inference: (\w+)",
                        server_log,
                    )
                )
            ),
        }
    assert len(cases) == 6 * manifest["repeats"]
    return manifest, cases, helper


def pool(manifest, cases):
    folder = OUT / "pooled"
    folder.mkdir(parents=True, exist_ok=True)
    paths, summary = {}, {}
    for tp in (1, 2):
        summary[str(tp)] = {}
        for arm in bench.ARMS:
            members = [
                cases[tp, arm, repeat] for repeat in range(1, manifest["repeats"] + 1)
            ]
            raw = copy.deepcopy(members[0]["raw"])
            raw["pooling"] = {
                "repetitions": manifest["repeats"],
                "method": "all successful request records, including warmup/cooldown",
                "source_reports": [str(m["report"]) for m in members],
            }
            values = {}
            for benchmark in raw["benchmarks"]:
                streams = int(benchmark["config"]["strategy"]["streams"])
                records = []
                for repeat, member in enumerate(members, 1):
                    current = next(
                        b
                        for b in member["raw"]["benchmarks"]
                        if int(b["config"]["strategy"]["streams"]) == streams
                    )
                    for record in current["requests"]["successful"]:
                        record = copy.deepcopy(record)
                        record["info"]["source_repeat"] = repeat
                        records.append(record)
                assert len(records) == manifest["repeats"] * 8 * streams
                benchmark["requests"] = {
                    "successful": records,
                    "errored": [],
                    "incomplete": [],
                }
                # The derived file is for request-level helpers, not GuideLLM metrics.
                benchmark["metrics"] = {}
                values[str(streams)] = {}
                for field in FIELDS:
                    pooled = statistics.mean(float(r[field]) for r in records)
                    runs = [member["means"][streams][field] for member in members]
                    assert math.isclose(pooled, statistics.mean(runs), rel_tol=1e-12)
                    values[str(streams)][field] = {
                        "mean": pooled,
                        "run_means": runs,
                        "minimum": min(runs),
                        "maximum": max(runs),
                        "stdev": statistics.stdev(runs),
                    }
            path = folder / f"tp{tp}-{arm}.json"
            bench.write(path, raw)
            paths[tp, arm] = path
            summary[str(tp)][arm] = values
    return paths, summary


def ratios(summary, repeats):
    output = {}
    for tp in ("1", "2"):
        output[tp] = {}
        for baseline, candidate in (
            ("rowwise", "triton-untuned"),
            ("rowwise", "triton-tuned"),
            ("triton-untuned", "triton-tuned"),
        ):
            key = f"{candidate}-vs-{baseline}"
            values = {}
            for streams in ("1", "2", "4", "8"):
                values[streams] = {}
                for field in FIELDS:
                    a, b = (
                        summary[tp][baseline][streams][field],
                        summary[tp][candidate][streams][field],
                    )
                    values[streams][field] = {
                        "ratio": b["mean"] / a["mean"],
                        "percent_change": (b["mean"] / a["mean"] - 1) * 100,
                        "paired_run_ratios": [
                            b["run_means"][i] / a["run_means"][i]
                            for i in range(repeats)
                        ],
                    }
            output[tp][key] = {
                "by_stream": values,
                "tps_geomean": geomean(
                    values[s]["output_tokens_per_second"]["ratio"] for s in values
                ),
                "e2e_geomean": geomean(
                    values[s]["request_latency"]["ratio"] for s in values
                ),
                "paired_run_tps_geomeans": [
                    geomean(
                        values[s]["output_tokens_per_second"]["paired_run_ratios"][i]
                        for s in values
                    )
                    for i in range(repeats)
                ],
            }
    return output


def overview(summary, field, filename, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for tp, ax in zip(("1", "2"), axes):
        for i, arm in enumerate(bench.ARMS):
            values = [summary[tp][arm][str(s)][field] for s in (1, 2, 4, 8)]
            means = np.array([v["mean"] for v in values])
            error = np.array(
                [
                    [v["mean"] - v["minimum"] for v in values],
                    [v["maximum"] - v["mean"] for v in values],
                ]
            )
            bars = ax.bar(
                np.arange(4) + (i - 1) * 0.25,
                means,
                0.25,
                yerr=error,
                capsize=3,
                label=LABELS[arm],
                color=COLORS[arm],
            )
            fmt = "%.0f" if field == "time_to_first_token_ms" else "%.1f"
            ax.bar_label(bars, fmt=fmt, padding=4, fontsize=8)
        ax.set_xticks(np.arange(4), ["1", "2", "4", "8"])
        ax.set_xlabel("Concurrent streams")
        ax.set_ylabel(ylabel)
        ax.set_title(f"TP{tp}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Gemma 4 12B FP8 · simple-agent multiturn · three fresh-server repetitions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def main():
    manifest, cases, helper = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    paths, summary = pool(manifest, cases)
    comparisons = ratios(summary, manifest["repeats"])
    concurrency = {}
    for tp in ("1", "2"):
        concurrency[tp] = {}
        for arm in bench.ARMS:
            concurrency[tp][arm] = {}
            for streams in (1, 2, 4, 8):
                runs = [
                    next(
                        c
                        for c in cases[int(tp), arm, rep]["success"]["counts"]
                        if c["streams"] == streams
                    )
                    for rep in range(1, manifest["repeats"] + 1)
                ]
                concurrency[tp][arm][str(streams)] = {
                    "mean": statistics.mean(c["mean_concurrency"] for c in runs),
                    "minimum": min(c["mean_concurrency"] for c in runs),
                    "maximum": max(c["mean_concurrency"] for c in runs),
                    "peak_by_run": [c["peak_concurrency"] for c in runs],
                }
    evidence = {
        f"tp{tp}-{arm}-rep{rep}": {
            "path": str(case["report"]),
            "sha256": bench.sha(case["report"]),
            "selected": case["success"]["selected"],
            "counts": case["success"]["counts"],
            "full_graph_capture_batch_sizes": case["full_graph_capture_batch_sizes"],
            "post_startup_jit_kernel_names": case["jit_kernel_names"],
            "prompt_tokens": case["prompt_tokens"],
        }
        for (tp, arm, rep), case in sorted(cases.items())
    }
    bench.write(
        OUT / "summary.json",
        {
            "head": bench.HEAD,
            "repeats": manifest["repeats"],
            "metrics": summary,
            "comparisons": comparisons,
            "concurrency": concurrency,
            "sources": evidence,
        },
    )
    with (OUT / "comparison.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "tp",
                "arm",
                "streams",
                "field",
                "pooled_mean",
                "run_min",
                "run_max",
                "run_stdev",
            ]
        )
        for tp, arms in summary.items():
            for arm, levels in arms.items():
                for streams, fields in levels.items():
                    for field, values in fields.items():
                        writer.writerow(
                            [
                                tp,
                                arm,
                                streams,
                                field,
                                values["mean"],
                                values["minimum"],
                                values["maximum"],
                                values["stdev"],
                            ]
                        )
    for field, filename, ylabel in (
        (
            "output_tokens_per_second",
            "serving_tps.png",
            "Mean per-request output tokens/s ↑",
        ),
        (
            "time_to_first_token_ms",
            "serving_ttft.png",
            "Mean time to first token (ms) ↓",
        ),
        ("request_latency", "serving_e2e.png", "Mean request latency (s) ↓"),
    ):
        overview(summary, field, filename, ylabel)
    text = "# Native per-token FP8: TP1 and TP2 serving comparisons\n\n"
    text += f"Source commit: `{bench.HEAD}`. Radeon AI PRO R9700 (gfx1201), Torch 2.13.0+rocm10, HIP 7.15.26333, Triton 3.8.0. Model `{bench.MODEL}` pinned to `{bench.REVISION}`. TP1 uses GPU0; TP2 uses GPUs 0 and 1. Attention: `ROCM_AITER_UNIFIED_ATTN`; FP8 KV cache, BF16 activations, prefix caching, max model length 32768, max batched tokens 8192, max sequences 8, GPU memory utilization 0.90.\n\n"
    text += "Three arms: Torch RowWise (the previous default, selected with `--linear-backend torch`), native Triton heuristic (same source with all per-token tuned maps removed in a separate worktree), and native Triton with the checked-in maps (`--linear-backend triton`). AITER linear is disabled in every arm; AITER attention remains enabled. Server logs verify the selected implementation and full GPU graph capture. The tuned arm loads six maps for each TP setting; the other arms load none.\n\n"
    text += "Workload: eight-turn simple-agent conversations, 2048 synthetic prompt tokens and exactly 512 generated output tokens per turn; streams 1/2/4/8; seed 20260715; one configured warmup request. Three fresh-server repetitions per TP/arm, with separate empty compilation caches and cyclic arm order. Every accepted run has 8/16/32/64 successful turn records, all eight turns for each conversation, and zero errors or incomplete requests. No performance-based run selection is used.\n\n"
    prompt_delta = max(
        abs(
            cases[tp, arm, rep]["prompt_tokens"][streams]["mean"]
            / cases[tp, "rowwise", rep]["prompt_tokens"][streams]["mean"]
            - 1
        )
        * 100
        for tp in (1, 2)
        for arm in bench.ARMS[1:]
        for rep in range(1, manifest["repeats"] + 1)
        for streams in ("1", "2", "4", "8")
    )
    text += f"Conversation history increases the full prompt from approximately 2k to 20k tokens. Later-turn prompt lengths can vary because generated responses are retokenized as history; the maximum absolute difference in mean full prompt length between a native arm and RowWise within a matched repetition/profile was {prompt_delta:.3f}%. Actual prompt-token ranges and means are retained in `summary.json` with each source report. Generated output usage is exactly 512 tokens for every accepted request.\n\n"
    text += "Client: GuideLLM 0.7.3, with `GUIDELLM__MAX_WORKER_PROCESSES=1` and `GUIDELLM__MP_MAX_PENDING_BUFFER_PERCENT=2.0` for every arm. The single worker schedules up to eight HTTP requests asynchronously, avoiding uneven conversation assignment from the shared multi-worker queue. A separate CPU mock test reached peak concurrency 1/2/4/8 and the expected request counts with these settings; multiple workers underfilled the targets. Every accepted GPU profile must also reach its configured peak. The earlier multi-worker pilots are excluded. Mean achieved concurrency is calculated as the sum of request latencies divided by the request window, including gaps between turns; it is reported below.\n\n"
    text += "Tables and plots use all successful request records, including warmup/cooldown, rather than GuideLLM aggregate metrics. TPS is per request. Bars show pooled means; error bars show the minimum and maximum of the three independent repetition means. Higher TPS and lower TTFT/E2E latency improve performance. Geometric means weight the four concurrency levels equally. These are end-to-end serving measurements.\n\n"
    text += "Fresh caches can leave new prefill launch variants to compile on their first use, even after full decode graph capture. The first accepted untuned TP1 repetition logged a `scaled_mm_kernel` compilation during its first measured prefill. Any first-request costs remain in the all-request metrics and per-turn plots. The raw server logs and numeric summary retain the names of kernels with post-startup JIT warnings; the monitor reports each kernel name once, so these names are not counts of all compilation events.\n\n"
    text += "| TP | Untuned Triton vs RowWise, TPS GM | Tuned Triton vs RowWise, TPS GM | Tuning vs heuristic, TPS GM |\n|---:|---:|---:|---:|\n"
    for tp in ("1", "2"):
        values = [
            comparisons[tp][name]["tps_geomean"]
            for name in (
                "triton-untuned-vs-rowwise",
                "triton-tuned-vs-rowwise",
                "triton-tuned-vs-triton-untuned",
            )
        ]
        text += (
            f"| {tp} | "
            + " | ".join(f"{v:.3f}x ({(v - 1) * 100:+.1f}%)" for v in values)
            + " |\n"
        )
    text += "\nPaired-repetition TPS geometric mean ranges (all three repetitions retained):\n\n| TP | Untuned vs RowWise | Tuned vs RowWise | Tuned vs untuned |\n|---:|---:|---:|---:|\n"
    for tp in ("1", "2"):
        text += (
            f"| {tp} | "
            + " | ".join(
                f"{min(comparisons[tp][name]['paired_run_tps_geomeans']):.3f}–{max(comparisons[tp][name]['paired_run_tps_geomeans']):.3f}x"
                for name in (
                    "triton-untuned-vs-rowwise",
                    "triton-tuned-vs-rowwise",
                    "triton-tuned-vs-triton-untuned",
                )
            )
            + " |\n"
        )
    text += "\n| TP | Streams | RowWise TPS | Untuned Triton TPS | Tuned Triton TPS | Untuned vs RowWise | Tuned vs RowWise |\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for tp in ("1", "2"):
        for streams in ("1", "2", "4", "8"):
            values = [
                summary[tp][arm][streams]["output_tokens_per_second"]["mean"]
                for arm in bench.ARMS
            ]
            text += (
                f"| {tp} | {streams} | "
                + " | ".join(f"{v:.2f}" for v in values)
                + " | "
                + " | ".join(
                    helper.format_change((v / values[0] - 1) * 100, False)
                    for v in values[1:]
                )
                + " |\n"
            )
    for filename in ("serving_tps.png", "serving_ttft.png", "serving_e2e.png"):
        text += f"\n![{filename}]({filename})\n"
    text += "\nMean achieved concurrency across repetitions (all runs reach the configured peak):\n\n| TP | Configured streams | RowWise | Untuned Triton | Tuned Triton |\n|---:|---:|---:|---:|---:|\n"
    for tp in ("1", "2"):
        for streams in ("1", "2", "4", "8"):
            text += (
                f"| {tp} | {streams} | "
                + " | ".join(
                    f"{concurrency[tp][arm][streams]['mean']:.3f}" for arm in bench.ARMS
                )
                + " |\n"
            )
    for tp in (1, 2):
        for baseline, candidate in (
            ("rowwise", "triton-untuned"),
            ("rowwise", "triton-tuned"),
            ("triton-untuned", "triton-tuned"),
        ):
            pair = OUT / f"tp{tp}-{candidate}-vs-{baseline}"
            subprocess.run(
                [
                    str(bench.BENCH / ".venv/bin/python"),
                    str(
                        bench.BENCH
                        / "guidellm-benchmark-compare/scripts/plot_compare_multiturn.py"
                    ),
                    "--report",
                    f"{LABELS[baseline]}={paths[tp, baseline]}",
                    "--report",
                    f"{LABELS[candidate]}={paths[tp, candidate]}",
                    "--title",
                    f"Gemma 4 12B FP8 · TP{tp} · three repetitions",
                    "--output-dir",
                    str(pair),
                ],
                check=True,
            )
            table = subprocess.check_output(
                [
                    str(bench.BENCH / ".venv/bin/python"),
                    str(
                        bench.BENCH
                        / "guidellm-benchmark-compare/scripts/summarize_guidellm.py"
                    ),
                    str(paths[tp, baseline]),
                    str(paths[tp, candidate]),
                ],
                text=True,
            )
            details = f"# TP{tp}: {LABELS[baseline]} → {LABELS[candidate]}\n\nPooled successful requests from three matched repetitions per arm; see the [full setup and overview](../report.md). Positive TPS changes and negative TTFT/E2E changes improve performance.\n\n{table}\n"
            for filename in (
                "compare_tps_by_turn.png",
                "compare_ttft_by_turn.png",
                "compare_e2e_by_turn.png",
            ):
                assert (pair / filename).stat().st_size > 1000
                details += f"\n![{filename}]({filename})\n"
            (pair / "report.md").write_text(details)
            text += f"\n[TP{tp}: {LABELS[baseline]} → {LABELS[candidate]}]({pair.name}/report.md)\n"
    text += "\nThree repetition means quantify observed run variability; no claim of statistical significance is made. Model-accuracy evaluation and hardware validation beyond gfx1201 remain separate pending work. Raw accepted reports, launch settings, source/config hashes, and derived request-level data are retained with this comparison.\n"
    (OUT / "report.md").write_text(text)
    for figure in OUT.glob("*.png"):
        assert figure.stat().st_size > 1000
    bench.status(
        "comparisons-complete", report=str(OUT / "report.md"), comparisons=comparisons
    )
    print(
        json.dumps(
            {
                "report": str(OUT / "report.md"),
                "comparisons": {
                    tp: {name: row["tps_geomean"] for name, row in values.items()}
                    for tp, values in comparisons.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
