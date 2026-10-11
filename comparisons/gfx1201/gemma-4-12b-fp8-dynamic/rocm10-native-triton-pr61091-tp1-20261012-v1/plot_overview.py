"""Plot absolute serving means separately for each same-GPU comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = {
    "tps": (
        "output_tokens_per_second",
        "Mean per-request output rate (tokens/s)",
        "Output rate",
    ),
    "ttft": (
        "time_to_first_token_ms",
        "Mean time to first token (ms)",
        "Time to first token",
    ),
    "e2e": ("request_latency", "Mean request latency (s)", "End-to-end latency"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    data = json.loads((args.comparison_dir / "summary.json").read_text())
    for metric, (field, ylabel, title) in METRICS.items():
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for axis, pair in zip(axes, data["pairs"], strict=True):
            for arm, label, color in (
                ("rowwise", "Torch RowWise", "#3366aa"),
                ("triton-tuned", "Native Triton", "#d9791f"),
            ):
                means = pair["cases"][arm]
                streams = sorted(map(int, means))
                axis.plot(
                    range(len(streams)),
                    [means[str(s)][field] for s in streams],
                    color=color,
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    label=label,
                )
            repeat = pair["repeat"]
            order = "Torch then Triton" if repeat == 1 else "Triton then Torch"
            axis.set_title(
                f"Comparison {repeat}: {order}\n(same GPU for both backends)",
                fontsize=11,
            )
            axis.set_xticks(range(len(streams)), streams)
            axis.set_xlabel("Concurrent streams")
            axis.grid(alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
        upper = max(
            values[field]
            for pair in data["pairs"]
            for means in pair["cases"].values()
            for values in means.values()
        )
        axes[0].set_ylim(0, upper * 1.10)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.87),
            ncols=2,
            frameon=False,
        )
        figure.suptitle(f"Gemma 4 12B FP8, TP1 on R9700 — {title}", fontsize=14, y=0.98)
        figure.text(
            0.5,
            0.015,
            "All successful requests included; absolute values kept separate for each GPU.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.83))
        path = args.comparison_dir / f"serving_{metric}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
