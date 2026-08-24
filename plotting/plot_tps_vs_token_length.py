"""Plot request output throughput against input and output token length.

The input is a GuideLLM JSON report. Each benchmark in the report becomes one
line, identified by its configured concurrency (``streams``). Request-level
points are averaged when multiple requests have the same token length.

Example:
    python plotting/plot_tps_vs_token_length.py \
        results/gfx1151/agent-multiturn-benchmarks.json \
        --output plots/tps_vs_token_length.png
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
    """Return finite numeric values while treating missing values as unusable."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _request_value(request: dict[str, Any], field: str) -> float | None:
    """Read a computed request field, with a small compatibility fallback."""
    value = _number(request.get(field))
    if value is not None:
        return value

    if field == "prompt_tokens":
        return _number(request.get("input_metrics", {}).get("total_tokens"))
    if field == "output_tokens":
        return _number(request.get("output_metrics", {}).get("total_tokens"))
    if field == "output_tokens_per_second":
        output_tokens = _request_value(request, "output_tokens")
        latency = _number(request.get("request_latency"))
        if output_tokens is not None and latency and latency > 0:
            return output_tokens / latency
    return None


def load_points(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load successful request points from a GuideLLM report."""
    with path.open() as file:
        report = json.load(file)

    labels = report.get("config", {}).get("metadata", {}).get("labels", {})
    workload = labels.get("workload") or path.stem
    points: list[dict[str, Any]] = []

    for benchmark in report.get("benchmarks", []):
        strategy = benchmark.get("config", {}).get("strategy", {})
        concurrency = _number(strategy.get("streams"))
        if concurrency is None:
            concurrency = _number(strategy.get("max_concurrency"))
        if concurrency is None:
            continue
        concurrency = int(concurrency)

        requests = benchmark.get("requests", {}).get("successful", [])
        for request in requests:
            prompt_tokens = _request_value(request, "prompt_tokens")
            output_tokens = _request_value(request, "output_tokens")
            output_tps = _request_value(request, "output_tokens_per_second")
            if None in (prompt_tokens, output_tokens, output_tps):
                continue
            points.append(
                {
                    "concurrency": concurrency,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "output_tokens_per_second": output_tps,
                }
            )

    return str(workload), points


def _aggregate(
    points: list[dict[str, Any]],
    x_field: str,
    group_tolerance: float,
    reducer: str = "mean",
) -> dict[int, list[tuple[float, float]]]:
    """Reduce request TPS values within nearby token-length buckets."""
    if reducer not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reducer: {reducer}")
    by_concurrency: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in points:
        by_concurrency[point["concurrency"]].append(
            (point[x_field], point["output_tokens_per_second"])
        )

    aggregated: dict[int, list[tuple[float, float]]] = {}
    for concurrency, values in by_concurrency.items():
        values.sort()
        clusters: list[list[tuple[float, float]]] = []
        for value in values:
            if not clusters or value[0] - clusters[-1][-1][0] > group_tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)

        aggregated[concurrency] = [
            (
                sum(x_value for x_value, _ in cluster) / len(cluster),
                (
                    sum(tps for _, tps in cluster)
                    if reducer == "sum"
                    else sum(tps for _, tps in cluster) / len(cluster)
                ),
            )
            for cluster in clusters
        ]
    return aggregated


def _plot_panel(
    axis: plt.Axes,
    points: list[dict[str, Any]],
    x_field: str,
    x_label: str,
    group_tolerance: float,
) -> None:
    grouped = _aggregate(points, x_field, group_tolerance)
    colors = plt.get_cmap("viridis", max(len(grouped), 1))

    for index, concurrency in enumerate(sorted(grouped)):
        x_values, tps_values = zip(*grouped[concurrency])
        axis.plot(
            x_values,
            tps_values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors(index),
            label=f"Concurrency {concurrency}",
        )

    axis.set_title(f"TPS vs {x_label.lower()}")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean per-request output TPS (tokens/s)")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Configured streams", fontsize=9)

    if not grouped:
        axis.text(
            0.5,
            0.5,
            "No successful request data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return

    if all(len(values) == 1 for values in grouped.values()):
        axis.text(
            0.02,
            0.03,
            "Only one observed length per concurrency",
            fontsize=8,
            color="dimgray",
            transform=axis.transAxes,
        )


def _plot_aggregate_panel(
    axis: plt.Axes,
    points: list[dict[str, Any]],
    group_tolerance: float,
) -> None:
    """Plot the sum of request TPS values within each input-length bucket."""
    input_groups = _aggregate(
        points,
        "prompt_tokens",
        group_tolerance,
        reducer="sum",
    )
    colors = plt.get_cmap("viridis", max(len(input_groups), 1))

    for index, concurrency in enumerate(sorted(input_groups)):
        x_values, aggregate_values = zip(*input_groups[concurrency])
        axis.plot(
            x_values,
            aggregate_values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors(index),
            label=f"Concurrency {concurrency}",
        )

    axis.set_title("Concurrency-summed TPS vs input / prompt tokens")
    axis.set_xlabel("Input / prompt tokens")
    axis.set_ylabel("Summed per-request output TPS (tokens/s)")
    axis.grid(True, alpha=0.25)
    axis.legend(
        title="Configured streams",
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )


def plot_report(
    input_path: Path,
    output_path: Path,
    dpi: int = 150,
    group_tolerance: float = 256.0,
) -> None:
    workload, points = load_points(input_path)
    if not points:
        raise ValueError(f"No successful request points found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle(
        f"Per-request and concurrency-summed output TPS — {workload}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    figure.subplots_adjust(bottom=0.2, top=0.84, right=0.8, wspace=0.25)
    _plot_panel(
        axes[0],
        points,
        "prompt_tokens",
        "Input / prompt tokens",
        group_tolerance,
    )
    _plot_aggregate_panel(axes[1], points, group_tolerance)
    figure.text(
        0.5,
        0.03,
        f"Left: mean request.output_tokens_per_second | Right: sum(request.output_tokens_per_second) within each input-length/concurrency bucket | Source: {input_path.name} | Nearby input lengths within {group_tolerance:g} tokens are averaged",
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
        default=Path("plots/tps_vs_token_length.png"),
        help="PNG output path (default: plots/tps_vs_token_length.png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution")
    parser.add_argument(
        "--group-tolerance",
        type=float,
        default=256.0,
        help="Group nearby token lengths before averaging (default: 256)",
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
