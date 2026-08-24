"""Plot TTFT against input token length for each concurrency level.

The left panel shows mean request TTFT. The right panel shows p95 request TTFT.
TTFT is a latency metric, so it is not summed across concurrent requests.

Example:
    python plotting/plot_ttft_vs_token_length.py \
        results/gfx1151/agent-multiturn-benchmarks.json \
        --output plots/ttft_vs_token_length.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def load_points(path: Path) -> tuple[str, list[dict[str, float | int]]]:
    """Load successful request TTFT points from a GuideLLM JSON report."""
    with path.open() as file:
        report = json.load(file)

    labels = report.get("config", {}).get("metadata", {}).get("labels", {})
    workload = labels.get("workload") or path.stem
    points: list[dict[str, float | int]] = []

    for benchmark in report.get("benchmarks", []):
        strategy = benchmark.get("config", {}).get("strategy", {})
        concurrency = _number(strategy.get("streams"))
        if concurrency is None:
            concurrency = _number(strategy.get("max_concurrency"))
        if concurrency is None:
            continue

        for request in benchmark.get("requests", {}).get("successful", []):
            prompt_tokens = _number(request.get("prompt_tokens"))
            ttft_ms = _number(request.get("time_to_first_token_ms"))
            if prompt_tokens is None:
                prompt_tokens = _number(
                    request.get("input_metrics", {}).get("total_tokens")
                )
            if ttft_ms is None:
                timings = request.get("info", {}).get("timings", {})
                request_start = _number(timings.get("request_start"))
                first_token = _number(timings.get("first_token_iteration"))
                if request_start is not None and first_token is not None:
                    ttft_ms = (first_token - request_start) * 1000
            if prompt_tokens is None or ttft_ms is None:
                continue
            points.append(
                {
                    "concurrency": int(concurrency),
                    "prompt_tokens": prompt_tokens,
                    "ttft_ms": ttft_ms,
                }
            )

    return str(workload), points


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _aggregate(
    points: list[dict[str, float | int]],
    group_tolerance: float,
    statistic: str,
) -> dict[int, list[tuple[float, float]]]:
    by_concurrency: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in points:
        by_concurrency[int(point["concurrency"])].append(
            (float(point["prompt_tokens"]), float(point["ttft_ms"]))
        )

    result: dict[int, list[tuple[float, float]]] = {}
    for concurrency, values in by_concurrency.items():
        values.sort()
        clusters: list[list[tuple[float, float]]] = []
        for value in values:
            if not clusters or value[0] - clusters[-1][-1][0] > group_tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)

        reduced: list[tuple[float, float]] = []
        for cluster in clusters:
            x_value = sum(x for x, _ in cluster) / len(cluster)
            ttft_values = [ttft for _, ttft in cluster]
            if statistic == "mean":
                y_value = sum(ttft_values) / len(ttft_values)
            elif statistic == "p95":
                y_value = _percentile(ttft_values, 0.95)
            else:
                raise ValueError(f"Unsupported TTFT statistic: {statistic}")
            reduced.append((x_value, y_value))
        result[concurrency] = reduced
    return result


def _plot_panel(
    axis: plt.Axes,
    grouped: dict[int, list[tuple[float, float]]],
    title: str,
    statistic: str,
    colors: Any,
) -> None:
    for index, concurrency in enumerate(sorted(grouped)):
        x_values, y_values = zip(*grouped[concurrency])
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors(index),
            label=f"Concurrency {concurrency}",
        )

    axis.set_title(title)
    axis.set_xlabel("Input / prompt tokens")
    axis.set_ylabel(f"{statistic} TTFT (ms)")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Configured streams", fontsize=9)


def plot_report(
    input_path: Path,
    output_path: Path,
    dpi: int = 150,
    group_tolerance: float = 256.0,
) -> None:
    workload, points = load_points(input_path)
    if not points:
        raise ValueError(f"No successful TTFT request points found in {input_path}")

    mean_groups = _aggregate(points, group_tolerance, "mean")
    p95_groups = _aggregate(points, group_tolerance, "p95")
    colors = plt.get_cmap("viridis", max(len(mean_groups), 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle(
        f"TTFT by input length — {workload}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    figure.subplots_adjust(bottom=0.2, top=0.84, wspace=0.25)
    _plot_panel(
        axes[0],
        mean_groups,
        "Mean TTFT vs input / prompt tokens",
        "Mean",
        colors,
    )
    _plot_panel(
        axes[1],
        p95_groups,
        "p95 TTFT vs input / prompt tokens",
        "p95",
        colors,
    )
    figure.text(
        0.5,
        0.03,
        f"GuideLLM metric: request.time_to_first_token_ms | TTFT is not summed across users | Nearby input lengths within {group_tolerance:g} tokens are averaged | Source: {input_path.name}",
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="GuideLLM benchmarks.json report")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/ttft_vs_token_length.png"),
        help="PNG output path (default: plots/ttft_vs_token_length.png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution")
    parser.add_argument(
        "--group-tolerance",
        type=float,
        default=256.0,
        help="Group nearby input lengths before calculating statistics (default: 256)",
    )
    args = parser.parse_args()
    plot_report(
        args.input,
        args.output,
        dpi=args.dpi,
        group_tolerance=args.group_tolerance,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
