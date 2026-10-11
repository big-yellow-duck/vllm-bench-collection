"""Validate and summarize the two same-GPU Torch/Triton serving comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SUITE = "rocm10-native-triton-pr61091-tp1-20261012-v1"
MODEL = "RedHatAI/gemma-4-12B-it-FP8-Dynamic"
HEAD = "678e155d2b7f1a63702dbed1025879ddd5a88bcc"
STREAMS = (1, 2, 4, 8)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def normalized_spec(spec):
    value = json.loads(json.dumps(spec))
    value["backend"].pop("target", None)
    value.pop("outputs", None)
    return value


def validate(path):
    data = read(path)
    assert data["metadata"]["guidellm_version"] == "0.7.3"
    spec = data["config"]["spec"]
    assert spec["profile"]["streams"] == list(STREAMS)
    assert spec["seed"] == {"kind": "static", "value": 20260715}
    assert spec["tokenizer"]["model"] == MODEL
    assert spec["data"][0]["turns"] == 8
    assert spec["data"][0]["prompt_tokens"] == 2048
    assert spec["data"][0]["output_tokens"] == 512
    assert spec["constraints"][0]["count"] == [8, 16, 32, 64]
    assert len(data["benchmarks"]) == 4
    seen, counts, prompts = set(), [], {}
    for benchmark in data["benchmarks"]:
        streams = int(benchmark["config"]["strategy"]["streams"])
        assert streams in STREAMS and streams not in seen
        seen.add(streams)
        assert benchmark["config"]["backend"]["model"] == MODEL
        assert not benchmark["config"]["environment"]["attributes"]["run_errors"]
        requests = benchmark["requests"]
        assert not requests.get("errored") and not requests.get("incomplete")
        successful = requests["successful"]
        assert len(successful) == 8 * streams
        conversations, per_turn = defaultdict(list), defaultdict(list)
        for request in successful:
            assert request["output_tokens"] == 512
            assert request["info"]["status"] == "completed"
            assert request["info"]["error"] is None
            assert request["request_end_time"] > request["request_start_time"]
            for field in (
                "output_tokens_per_second",
                "time_to_first_token_ms",
                "request_latency",
            ):
                assert math.isfinite(request[field]) and request[field] > 0
            conversations[request["info"]["conversation_id"]].append(
                request["info"]["turn_index"]
            )
            per_turn[request["info"]["turn_index"]].append(request["prompt_tokens"])
        assert len(conversations) == streams
        assert all(sorted(turns) == list(range(8)) for turns in conversations.values())
        events = sorted(
            (timestamp, delta)
            for request in successful
            for timestamp, delta in (
                (request["request_start_time"], 1),
                (request["request_end_time"], -1),
            )
        )
        active = peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        assert peak == streams
        window = max(r["request_end_time"] for r in successful) - min(
            r["request_start_time"] for r in successful
        )
        counts.append(
            {
                "streams": streams,
                "successful": len(successful),
                "errored": 0,
                "incomplete": 0,
                "peak_concurrency": peak,
                "mean_concurrency": sum(r["request_latency"] for r in successful)
                / window,
            }
        )
        prompts[streams] = {
            turn + 1: statistics.mean(values) for turn, values in per_turn.items()
        }
    return (
        normalized_spec(spec),
        sorted(counts, key=lambda row: row["streams"]),
        prompts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[4]
    )
    args = parser.parse_args()
    comparison = args.repo / "comparisons/gfx1201/gemma-4-12b-fp8-dynamic" / SUITE
    results = args.repo / "results/gfx1201/gemma-4-12b-fp8-dynamic" / SUITE
    spec = importlib.util.spec_from_file_location(
        "summary_helper", comparison / "scripts/summarize_guidellm.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    common_spec, common_command, common_env, common_runtime = None, None, None, None
    pairs, all_counts = [], []
    for repeat in (1, 2):
        row = {"repeat": repeat, "cases": {}, "mean_prompt_tokens_by_turn": {}}
        for arm in ("rowwise", "triton-tuned"):
            name = f"tp1-{arm}-rep{repeat}"
            report = results / name / "agent-multiturn-benchmarks.json"
            current_spec, counts, prompts = validate(report)
            if common_spec is None:
                common_spec = current_spec
            assert common_spec == current_spec, f"Workload mismatch: {name}"
            provenance = read(results / name / "provenance.json")
            assert (
                hashlib.sha256(report.read_bytes()).hexdigest()
                == provenance["report_sha256"]
            )
            assert provenance["vllm_commit"] == HEAD
            assert provenance["case"] == {"tp": 1, "arm": arm, "repeat": repeat}
            assert provenance["revision_arguments"] is False
            assert provenance["physical_gpu"] == str(repeat - 1)
            assert provenance["source_immutability_verified"] is True
            assert provenance["counts"] == counts
            command = list(provenance["portable_command"])
            backend_index = command.index("--linear-backend") + 1
            assert command[backend_index] == ("torch" if arm == "rowwise" else "triton")
            command[backend_index] = "<compared-backend>"
            if common_command is None:
                common_command, common_env, common_runtime = (
                    command,
                    provenance["env"],
                    provenance["runtime"],
                )
            assert command == common_command
            assert provenance["env"] == common_env
            assert provenance["runtime"] == common_runtime
            expected = (
                "RowWiseTorchFP8ScaledMMLinearKernel"
                if arm == "rowwise"
                else "TritonFp8PerTokenScaledMMKernel"
            )
            assert provenance["selected_linear_kernels"] == [expected]
            row["cases"][arm] = helper.load_means(report)
            row["mean_prompt_tokens_by_turn"][arm] = prompts
            all_counts.append({"case": name, "counts": counts})
        row["changes_percent"] = {
            streams: {
                field: helper.change(
                    row["cases"]["triton-tuned"][streams][field],
                    row["cases"]["rowwise"][streams][field],
                )
                for field, _, _ in helper.FIELDS
            }
            for streams in STREAMS
        }
        row["prompt_length_difference_percent_max"] = max(
            abs(
                helper.change(
                    row["mean_prompt_tokens_by_turn"]["triton-tuned"][streams][turn],
                    row["mean_prompt_tokens_by_turn"]["rowwise"][streams][turn],
                )
            )
            for streams in STREAMS
            for turn in range(1, 9)
        )
        pairs.append(row)
    paired = {
        streams: {
            field: (
                math.prod(
                    1 + pair["changes_percent"][streams][field] / 100 for pair in pairs
                )
                ** (1 / len(pairs))
                - 1
            )
            * 100
            for field, _, _ in helper.FIELDS
        }
        for streams in STREAMS
    }
    ranges = {
        streams: {
            field: [
                min(pair["changes_percent"][streams][field] for pair in pairs),
                max(pair["changes_percent"][streams][field] for pair in pairs),
            ]
            for field, _, _ in helper.FIELDS
        }
        for streams in STREAMS
    }
    validation = {
        "vllm_commit": HEAD,
        "successful_requests": 480,
        "errored_requests": 0,
        "incomplete_requests": 0,
        "workload_compatible": True,
        "method": "Means over all requests.successful, including first-use, warmup and cooldown requests. Geometric mean of two within-GPU candidate/baseline ratios; absolute values are never pooled across GPUs.",
        "counts": all_counts,
        "pairs": pairs,
        "paired_changes_percent": paired,
        "paired_change_ranges_percent": ranges,
    }
    write(comparison / "summary.json", validation)
    headers = [
        "Streams",
        "Output tok/s change",
        "TPS change range",
        "Mean TTFT change",
        "Mean E2E change",
    ]
    table = ["| " + " | ".join(headers) + " |", "|---:|---:|---:|---:|---:|"]
    for streams in STREAMS:
        low, high = ranges[streams]["output_tokens_per_second"]
        cells = [
            str(streams),
            helper.format_change(paired[streams]["output_tokens_per_second"], False),
            f"{low:+.1f}% to {high:+.1f}%",
        ]
        cells += [
            helper.format_change(paired[streams][field], lower)
            for field, _, lower in helper.FIELDS[1:]
        ]
        table.append("| " + " | ".join(cells) + " |")
    paired_table = "\n".join(table)
    (comparison / "paired-table.md").write_text(paired_table + "\n")
    text = [
        "# Native Triton per-token FP8 versus Torch RowWise",
        "",
        "Fresh TP1 serving comparisons for [vLLM PR #61091](https://github.com/vllm-project/vllm/pull/61091).",
        "",
        f"Measured vLLM commit: [`{HEAD[:12]}`](https://github.com/big-yellow-duck/vllm/commit/{HEAD}).",
        "",
        "## Setup",
        "",
        "Radeon AI PRO R9700 (gfx1201), Torch 2.13 with ROCm 10, Triton 3.8, GuideLLM 0.7.3. "
        f"Model: `{MODEL}`. Both backends use ROCM_AITER_UNIFIED_ATTN, FP8 KV cache, "
        "BF16 output, default compilation/graphs and prefix caching. AITER linear is disabled. "
        "Only `--linear-backend torch` versus `--linear-backend triton` changes.",
        "",
        "Two independent comparisons use two cards. Each baseline/candidate pair runs sequentially "
        "on the same physical GPU: pair 1 is Torch then Triton; pair 2 is Triton then Torch. "
        "Every configuration starts a fresh server with empty compilation caches. "
        "The first greeting smoke request is outside the benchmark; every successful benchmark request, "
        "including first-use specialization and configured warmup/cooldown requests, is included below. "
        "Server startup and graph capture are outside the request measurements.",
        "",
        "## Compatibility and completeness",
        "",
        "All four reports use the same model ID, tokenizer ID, synthetic workload and seed (20260715). "
        "Each profile has 8 turns, 2048 configured prompt tokens per turn, and exactly 512 output tokens "
        "per request. Streams 1/2/4/8 finish 8/16/32/64 requests respectively and attain the configured "
        "peak concurrency. All 480 requests succeed, with zero errors or incomplete requests. "
        "Server logs confirm the intended linear kernel, six tuning files for Triton, "
        "and completed graph capture. Tracked vLLM sources and benchmark runner hashes remain unchanged.",
        "",
        "Actual prompts include chat templates and prior conversation history. "
        "The x-axis shows mean observed prompt tokens. The largest within-pair difference in mean "
        f"prompt length is {max(p['prompt_length_difference_percent_max'] for p in pairs):.2f}%; "
        "generated history can differ between backends.",
        "",
        "## Paired results",
        "",
        "The table shows the geometric mean of the two within-GPU candidate/baseline ratios. "
        "The TPS range shows the two observed percentage changes, not a confidence interval. "
        "Absolute latencies and token rates are kept separate for each card. "
        "Positive output TPS changes and negative TTFT/E2E changes are improvements (bold). "
        "Output TPS is mean per-request end-to-end output tokens/s, including time to first token; "
        "it is not aggregate server throughput or a GEMM-only measurement.",
        "",
        paired_table,
        "",
        "![Output rate by concurrency](serving_tps.png)",
        "",
        "![Time to first token by concurrency](serving_ttft.png)",
        "",
        "![End-to-end request latency by concurrency](serving_e2e.png)",
        "",
    ]
    for pair in pairs:
        repeat = pair["repeat"]
        folder = comparison / f"pair{repeat}"
        pair_table = (folder / "table.md").read_text().strip()
        text += [
            f"## Pair {repeat}: same-GPU absolute results",
            "",
            pair_table,
            "",
            f"![Output TPS by turn, pair {repeat}](pair{repeat}/compare_tps_by_turn.png)",
            "",
            f"![TTFT by turn, pair {repeat}](pair{repeat}/compare_ttft_by_turn.png)",
            "",
            f"![E2E latency by turn, pair {repeat}](pair{repeat}/compare_e2e_by_turn.png)",
            "",
        ]
    text += [
        "## Interpretation and limits",
        "",
        f"Native Triton raises the paired mean per-request output rate by "
        f"{min(values['output_tokens_per_second'] for values in paired.values()):.1f}–"
        f"{max(values['output_tokens_per_second'] for values in paired.values()):.1f}% "
        "across the four stream levels. Both independent comparisons improve at every stream level. "
        "Mean E2E latency is lower at all four levels. "
        f"Single-stream mean TTFT increases by {paired[1]['time_to_first_token_ms']:.1f}%, "
        "with increases at turns 1 and 5 visible in both comparisons; mean TTFT falls at 2/4/8 streams. "
        "These turn-level costs are retained in the means and charts.",
        "",
        "See both per-pair tables and the per-turn charts when assessing repeatability. "
        "Two matched pairs are a small sample, and these results cover one model, TP1 and gfx1201. "
        "They do not establish performance on other hardware or model task accuracy. "
        "Fresh caches retain first-use effects; a TTFT regression must be read alongside output rate "
        "and total request latency rather than removed by excluding early requests.",
        "",
        "## Reproduce",
        "",
        "| Comparison | Torch RowWise report | Native Triton report |",
        "|---|---|---|",
        *[
            f"| {repeat} | [JSON](../../../../results/gfx1201/gemma-4-12b-fp8-dynamic/{SUITE}/tp1-rowwise-rep{repeat}/agent-multiturn-benchmarks.json) "
            f"| [JSON](../../../../results/gfx1201/gemma-4-12b-fp8-dynamic/{SUITE}/tp1-triton-tuned-rep{repeat}/agent-multiturn-benchmarks.json) |"
            for repeat in (1, 2)
        ],
        "",
        "Use the direct `vllm serve` commands in the PR, running the two configurations separately "
        "on the same GPU. Start each with an empty compilation cache. Then run:",
        "",
        "```bash",
        "GUIDELLM__MP_MAX_PENDING_BUFFER_PERCENT=2.0 \\",
        "GUIDELLM__MAX_WORKER_PROCESSES=1 \\",
        "guidellm run --config workload.json",
        "```",
        "",
        "The portable [workload](workload.json) targets `http://localhost:8000` and uses the model ID "
        "for tokenization. Neither serving command uses model or tokenizer revision arguments; "
        "the resolved default model revision is recorded in the [provenance](../../../../results/gfx1201/gemma-4-12b-fp8-dynamic/"
        + SUITE
        + "/provenance.json).",
        "",
        "Raw JSON reports and per-case provenance are under `results/gfx1201/gemma-4-12b-fp8-dynamic/"
        + SUITE
        + "`. "
        "Regenerate the charts with `scripts/plot_compare_multiturn.py` and the absolute result tables "
        "with `scripts/summarize_guidellm.py`, using the RowWise report as baseline. "
        "Run `analyze.py --repo <benchmark-repository>` to revalidate reports and regenerate the paired table. "
        "[summary.json](summary.json) contains the numeric source, completeness checks and prompt lengths.",
        "",
    ]
    (comparison / "README.md").write_text("\n".join(text))
    print(paired_table)
    print(
        "Validated 480 successful requests in four reports; zero errors or incomplete requests."
    )


if __name__ == "__main__":
    main()
