"""Prepare a verified AITER-only benchmark section for the current PR body."""

import argparse
import re
from pathlib import Path

import report_aiter_default_attn as report
import run_aiter_default_attn as bench

START = "<!-- aiter-default-serving-results:start -->"
END = "<!-- aiter-default-serving-results:end -->"


def render(body, commit):
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    results_heading = re.search(r"^### Results[^\S\r\n]*\r?$", body, re.MULTILINE)
    heading = "####" if results_heading else "###"
    manifest, cases, _ = report.collect()
    assert bench.read(bench.RUN / "status.json")["phase"] == "complete"
    assert len(cases) == 12 and manifest["repeats"] == 3
    data = bench.read(report.OUT / "summary.json")
    assert data["head"] == manifest["head"]
    relative = report.OUT.relative_to(bench.BENCH).as_posix()
    repository = "https://github.com/big-yellow-duck/vllm-bench-collection"
    documents = f"{repository}/blob/{commit}/{relative}"
    images = f"https://raw.githubusercontent.com/big-yellow-duck/vllm-bench-collection/{commit}/{relative}"
    text = f"{START}\n{heading} End-to-end serving: AITER linear off versus on\n\n"
    text += (
        f"Source `{manifest['head']}`; `{bench.MODEL}` with unpinned model/tokenizer "
        f"defaults resolving to `{manifest['revision']}` throughout the suite. "
        "One/two Radeon AI PRO R9700 GPUs (gfx1201), Torch 2.13.0+rocm10, Triton 3.8. "
        "Three fresh-server repetitions per TP/arm, 12 accepted runs. "
        "Full model loading; attention and prefix caching use their defaults. "
        "Gemma 4 selects `TRITON_ATTN`, and prefix caching enables itself. "
        "Both arms use `VLLM_ROCM_USE_AITER=1` and `--linear-backend auto`; "
        "only `VLLM_ROCM_USE_AITER_LINEAR` changes from 0 to 1. Logs verify "
        "Torch RowWise when off and AITER per-token FP8 when on. "
        "Installed tuning entries are recorded with hashes in the evidence; "
        "companion tuning PR: [ROCm/aiter#6245](https://github.com/ROCm/aiter/pull/6245).\n\n"
        "Eight-turn simple-agent conversations add 2048 synthetic prompt tokens "
        "and generate exactly 512 output tokens per turn at 1/2/4/8 streams "
        "(seed 20260715). BF16, FP8 KV cache, max model length 32768, "
        "max batched tokens 8192, max sequences 8, GPU memory utilization 0.90. "
        "Every profile reaches its configured peak and at least 98% mean "
        "concurrency, preserves all eight turns, and has zero errors/incomplete "
        "requests. Full decode graphs cover batches 1/2/4/8.\n\n"
        "| TP | Streams | RowWise tokens/s | AITER tokens/s | TPS change | TTFT change | E2E change |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    losses = []
    for tp in ("1", "2"):
        for streams in ("1", "2", "4", "8"):
            a, b = [
                data["metrics"][tp][arm][streams]["output_tokens_per_second"]["mean"]
                for arm in bench.ARMS
            ]
            changes = [
                data["comparisons"][tp][streams][field]["percent_change"]
                for field in report.FIELDS
            ]
            formatted = []
            for i, change in enumerate(changes):
                value = f"{change:+.1f}%"
                formatted.append(
                    f"**{value}**" if (change > 0 if i == 0 else change < 0) else value
                )
            text += (
                f"| {tp} | {streams} | {a:.2f} | {b:.2f} | "
                + " | ".join(formatted)
                + " |\n"
            )
            if changes[0] < 0:
                losses.append(f"TP{tp}, {streams} streams: {changes[0]:+.1f}% TPS")
    text += "\n| TP | TPS geometric-mean ratio, on/off | Paired repetition range |\n|---:|---:|---:|\n"
    for tp in ("1", "2"):
        comparison = data["comparisons"][tp]
        ratio = comparison["geomeans"][report.FIELDS[0]]
        runs = comparison["paired_tps_geomeans"]
        text += f"| {tp} | {ratio:.3f}x ({100 * (ratio - 1):+.1f}%) | {min(runs):.3f}–{max(runs):.3f}x |\n"
    if losses:
        text += "\nObserved TPS losses: " + "; ".join(losses) + ".\n"
    text += (
        "\nTPS is the mean per-request output rate; higher TPS and lower TTFT/E2E "
        "improve performance. Geomeans weight the four concurrency levels equally. "
        "Every successful request, including warmup/cooldown, is included. Error "
        "bars show min/max independent repetition means. Arm and TP order alternate. "
        "Each run uses empty compile caches; first-use prefill compilation remains "
        "included in these serving metrics. No performance-based run selection. "
        "GuideLLM 0.7.3 uses one asynchronous worker and pending buffer 2.0. "
        "These are synthetic-text serving measurements; accuracy and multimodal "
        "input validation remain separate. Historical runs are excluded.\n\n"
        f"[Full report, per-turn charts, raw results, and provenance]({documents}/report.md).\n\n"
    )
    for name, label in (
        ("serving_tps.png", "Output rate"),
        ("serving_ttft.png", "Time to first token"),
        ("serving_e2e.png", "Request latency"),
    ):
        assert (report.OUT / name).exists()
        text += f"![{label}]({images}/{name})\n\n"
    text += END
    if START in body:
        assert END in body
        return re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _: text,
            body,
            flags=re.DOTALL,
        )
    assert "## Validation" in body
    if results_heading:
        return (
            body[: results_heading.end()]
            + "\n\n"
            + text
            + "\n"
            + body[results_heading.end() :]
        )
    if "## Duplicate-work check" in body:
        return body.replace(
            "## Duplicate-work check", text + "\n\n## Duplicate-work check", 1
        )
    return body.rstrip() + "\n\n" + text + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--artifact-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.body.read_text(), args.artifact_commit))


if __name__ == "__main__":
    main()
