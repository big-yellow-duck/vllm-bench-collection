"""Compare multiple GuideLLM multi-turn reports one step at a time.

Each report/concurrency pair becomes one line. Reports whose labels contain a
tensor-parallel name such as TP2 or TP4 are split into separate charts. Comparing
baseline and FlyDSL-AR, each with streams 1, 2, 4, and 8, therefore produces
two lines in each concurrency subplot. Color identifies the report, and the
four subplots identify configured streams.

Example:
    python plotting/plot_compare_multiturn.py \
        --report "TP2 baseline=results/.../tp2-baseline/agent-multiturn-benchmarks.json" \
        --report "TP2 FlyDSL-AR=results/.../tp2-flydslar/agent-multiturn-benchmarks.json" \
        --output-dir plots/tp2-vs-tp4
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METRICS = {
    "tps": ("output_tokens_per_second", "Mean per-request output TPS (tokens/s)", "Output TPS"),
    "ttft": ("time_to_first_token_ms", "Mean TTFT (ms)", "Time to first token"),
    "e2e": ("request_latency", "Mean E2E latency (s)", "End-to-end latency"),
}
@dataclass(frozen=True)
class Report:
    label: str
    path: Path
    # concurrency -> [(turn, prompt tokens, metric values)]
    series: dict[int, dict[int, tuple[list[float], list[float]]]]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _parse_report(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("reports must use LABEL=PATH") from error
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("reports must use a non-empty LABEL=PATH")
    return label.strip(), Path(raw_path)


def load_report(label: str, path: Path) -> Report:
    with path.open() as file:
        data = json.load(file)

    series: dict[int, dict[int, tuple[list[float], list[float]]]] = {}
    for benchmark in data.get("benchmarks", []):
        strategy = benchmark.get("config", {}).get("strategy", {})
        concurrency_value = _number(strategy.get("streams"))
        if concurrency_value is None:
            concurrency_value = _number(strategy.get("max_concurrency"))
        if concurrency_value is None:
            continue
        concurrency = int(concurrency_value)
        turns: dict[int, tuple[list[float], list[float]]] = defaultdict(
            lambda: ([], [])
        )
        for request in benchmark.get("requests", {}).get("successful", []):
            turn_value = _number(request.get("info", {}).get("turn_index"))
            prompt_tokens = _number(request.get("prompt_tokens"))
            if prompt_tokens is None:
                prompt_tokens = _number(
                    request.get("input_metrics", {}).get("total_tokens")
                )
            if turn_value is None or prompt_tokens is None:
                continue
            turn = int(turn_value) + 1
            turns[turn][0].append(prompt_tokens)
            # Store all supported metric values in a fixed order.
            for metric_index, (_, (field, _, _)) in enumerate(METRICS.items()):
                while len(turns[turn][1]) < len(METRICS) * len(turns[turn][0]):
                    turns[turn][1].append(float("nan"))
                metric_value = _number(request.get(field))
                turns[turn][1][(len(turns[turn][0]) - 1) * len(METRICS) + metric_index] = (
                    metric_value if metric_value is not None else float("nan")
                )
        series[concurrency] = dict(turns)

    if not series:
        raise ValueError(f"No successful multi-turn request data found in {path}")
    return Report(label, path, series)


def _metric_points(
    report: Report, concurrency: int, metric: str
) -> tuple[list[int], list[float], list[float]]:
    metric_index = list(METRICS).index(metric)
    turns: list[int] = []
    prompt_lengths: list[float] = []
    values: list[float] = []
    for turn, (prompts, flattened_metrics) in sorted(report.series[concurrency].items()):
        metric_values = flattened_metrics[metric_index :: len(METRICS)]
        usable = [value for value in metric_values if value == value]
        if not usable:
            continue
        turns.append(turn)
        prompt_lengths.append(mean(prompts))
        values.append(mean(usable))
    return turns, prompt_lengths, values


def plot_metric(
    reports: list[Report],
    metric: str,
    output_path: Path,
    dpi: int,
    group_label: str,
) -> int:
    _, y_label, title = METRICS[metric]
    colors = plt.get_cmap("tab10")
    stream_values = sorted({value for report in reports for value in report.series})
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    line_count = 0
    for axis_index, concurrency in enumerate(stream_values):
        if axis_index >= len(flat_axes):
            raise ValueError("At most four concurrency levels can be plotted")
        axis = flat_axes[axis_index]
        prompt_lengths_by_turn: dict[int, list[float]] = defaultdict(list)
        for report_index, report in enumerate(reports):
            if concurrency not in report.series:
                continue
            turns, prompt_lengths, values = _metric_points(report, concurrency, metric)
            if not turns:
                continue
            axis.plot(
                turns,
                values,
                color=colors(report_index % 10),
                marker="o",
                linewidth=2,
                markersize=5,
                alpha=0.9,
                label=report.label,
            )
            line_count += 1
            for turn, prompt_length in zip(turns, prompt_lengths, strict=True):
                prompt_lengths_by_turn[turn].append(prompt_length)

        turns = sorted(prompt_lengths_by_turn)
        tick_labels = [
            f"{turn}\n{mean(prompt_lengths_by_turn[turn]) / 1000:.1f}k"
            for turn in turns
        ]
        axis.set_xticks(turns, tick_labels)
        axis.set_title(f"{concurrency} stream{'s' if concurrency != 1 else ''}")
        axis.grid(True, alpha=0.25)

    for axis in flat_axes[len(stream_values) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Conversation step\n(mean prompt tokens)")
    for axis in axes[:, 0]:
        axis.set_ylabel(y_label)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Result set",
        loc="upper center",
        ncols=max(len(labels), 1),
        bbox_to_anchor=(0.5, 0.95),
    )
    figure.suptitle(f"{group_label} — {title} by multi-turn step", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return line_count


def group_reports_by_tp(reports: list[Report]) -> dict[str, list[Report]]:
    """Group reports by a TP token in their label, preserving input order."""
    groups: dict[str, list[Report]] = {}
    ungrouped: list[Report] = []
    for report in reports:
        match = re.search(r"\bTP\d+\b", report.label, flags=re.IGNORECASE)
        if match is None:
            ungrouped.append(report)
            continue
        group = match.group(0).upper()
        groups.setdefault(group, []).append(report)

    if ungrouped:
        if groups:
            labels = ", ".join(report.label for report in ungrouped)
            raise ValueError(
                "Every report label must include a TP name when splitting "
                f"multiple TP groups; missing from: {labels}"
            )
        return {"comparison": ungrouped}
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        type=_parse_report,
        metavar="LABEL=PATH",
        help="Report to compare; repeat once per result set",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRICS),
        default=list(METRICS),
        help="Charts to produce (default: tps ttft e2e)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/comparison"),
        help="Directory for generated PNGs",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution")
    args = parser.parse_args()

    reports = [load_report(label, path) for label, path in args.report]
    report_groups = group_reports_by_tp(reports)
    for group_label, group_reports in report_groups.items():
        output_prefix = group_label.lower()
        for metric in args.metrics:
            output_path = (
                args.output_dir / f"compare_{output_prefix}_{metric}_by_turn.png"
            )
            line_count = plot_metric(
                group_reports,
                metric,
                output_path,
                args.dpi,
                group_label,
            )
            print(f"Wrote {output_path} ({line_count} lines)")


if __name__ == "__main__":
    main()
