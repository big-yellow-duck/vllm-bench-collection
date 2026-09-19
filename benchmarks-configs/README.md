# GuideLLM benchmark workflow

This guide uses GuideLLM 0.7.3 and the OpenAI-compatible server at
`http://localhost:8000`. The multi-turn workload is defined in
[`guidellm-agent-multiturn.json`](./guidellm-agent-multiturn.json).

## What this scenario measures

The scenario runs four fixed-concurrency benchmarks:

| Concurrent users | Conversations | Turns (HTTP requests) |
|---:|---:|---:|
| 1 | 1 | 8 |
| 2 | 2 | 16 |
| 4 | 4 | 32 |
| 8 | 8 | 64 |

Each conversation has eight user turns. Every turn adds approximately 2,048
new prompt tokens and asks for 512 output tokens. GuideLLM sends the previous
messages again on later turns, so the total input grows throughout a
conversation. Those repeated conversation prefixes exercise automatic prefix
caching when it is enabled on the inference server.

The synthetic workload is deterministic with seed `20260715`.

## Speculative-decoding prefix sweep

Do not use the synthetic scenario above by itself to evaluate speculative
decoding. Its generated prompts are synthetic, and GuideLLM 0.7.3 sends
`ignore_eos: true` when `output_tokens` is configured. That makes the fixed
2,048-in/512-out shape useful for engine and kernel comparisons, but it is not
a representative test of draft-token acceptance.

Use [`run-spec-decode-prefix-sweep.sh`](./run-spec-decode-prefix-sweep.sh) for
the speculative-decoding comparison. It uses real ShareGPT conversations and
resends the full generated conversation history on every turn, so each turn
extends the prefix cached by vLLM. The sweep keeps the original request budget:

| Concurrent conversations | Conversations | Turns (HTTP requests) |
|---:|---:|---:|
| 1 | 1 | 8 |
| 2 | 2 | 16 |
| 4 | 4 | 32 |
| 8 | 8 | 64 |

Responses are capped at 512 tokens, but EOS is respected. Prompt and output
lengths are therefore natural rather than forced; inspect the saved detailed
result to compare their distributions between runs. Sampling is greedy
(`temperature=0`) and the dataset seed is fixed so the same server configuration
produces a repeatable workload.

The script resets vLLM's prefix cache before every sweep point. This prevents a
conversation used at concurrency 1 from leaving cached KV blocks that make the
later concurrency points artificially fast. Start vLLM with
`VLLM_SERVER_DEV_MODE=1` so the reset endpoint is available. If that is not
possible, set `RESET_PREFIX_CACHE=0`; in that case, restart the server and run
each concurrency point separately for publishable comparisons.

On its first run, the script uses
[`prepare-sharegpt-prefix-sweep.py`](./prepare-sharegpt-prefix-sweep.py) to
stream a pinned revision of `Aeala/ShareGPT_Vicuna_unfiltered`, select only
conversations with eight valid user/assistant pairs, and cache a small local
benchmark file under `.benchmark-data/`. This works around a current
`vllm-bench` limitation: its ShareGPT loader honors the maximum turn count but
does not filter on the requested minimum. Set `DATASET_PATH` only when supplying
an already-filtered ShareGPT JSON file with exactly eight turns per conversation.

The maintained Rust `vllm-bench` client automatically snapshots vLLM's
speculative-decoding Prometheus counters before and after each run. Its output
therefore includes acceptance rate and mean acceptance length alongside
throughput, TTFT, TPOT, ITL, end-to-end latency, and per-turn metrics.
Run the benchmark against an otherwise idle server because these Prometheus
counters are server-wide.

Run it with defaults for the current Qwen server:

```bash
benchmarks-configs/run-spec-decode-prefix-sweep.sh
```

Override the endpoint, model, result directory, binary, or a pinned local
ShareGPT file without editing the script:

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=Qwen/Qwen3.8-27B-FP8 \
VLLM_BENCH_BIN=/path/to/vllm/rust/target/release/vllm-bench \
DATASET_PATH=/data/sharegpt-8turn.json \
RESULT_DIR=/app/results/spec-decode-prefix-sweep \
benchmarks-configs/run-spec-decode-prefix-sweep.sh
```

The default uses one conversation per concurrent stream, matching the original
8/16/32/64-request budget. For a less noisy characterization, set
`CONVERSATIONS_PER_STREAM=4` or higher; this deliberately increases the request
count at every sweep point.

For an apples-to-apples speculative-decoding comparison, run the sweep once
with speculative decoding disabled and once with it enabled. Restart the server
between configurations, keep all non-speculative server flags identical, and
keep the same prepared dataset.
Compare output-token throughput, TPOT/ITL, end-to-end latency, acceptance rate,
mean acceptance length, and error counts. Do not compare only acceptance rate:
a drafter can accept many tokens while still losing end-to-end performance to
drafting and verification overhead.

## Run into a chosen output directory

GuideLLM does not have one global `--output-dir` option. Set the `path` of each
requested output format instead. A convenient shell workflow is:

```bash
RESULTS_DIR="/app/results/agent-multiturn-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS_DIR"

guidellm run \
  --config /app/guidellm-agent-multiturn.json \
  --output "kind=json,path=${RESULTS_DIR}/benchmarks.json" \
  --output "kind=csv,path=${RESULTS_DIR}/benchmarks.csv" \
  --output "kind=html,path=${RESULTS_DIR}/benchmarks.html"
```

The CLI `--output` arguments override the output list in the scenario file.
Always retain `benchmarks.json`: it is the complete machine-readable report
from which the other formats can be regenerated.

If a stable directory is preferable to timestamped runs:

```bash
RESULTS_DIR=/app/results/agent-multiturn
mkdir -p "$RESULTS_DIR"

guidellm run \
  --config /app/guidellm-agent-multiturn.json \
  --output "kind=json,path=${RESULTS_DIR}/benchmarks.json" \
  --output "kind=csv,path=${RESULTS_DIR}/benchmarks.csv" \
  --output "kind=html,path=${RESULTS_DIR}/benchmarks.html"
```

This stable-directory form overwrites files from a previous run, so use the
timestamped form when retaining benchmark history matters.

## Produce readable reports from an existing JSON result

Re-exporting is fast and does not rerun inference. For the completed result in
this workspace:

```bash
SOURCE_JSON=/app/agent-multiturn-benchmarks.json
RESULTS_DIR=/app/results/agent-multiturn-report
mkdir -p "$RESULTS_DIR"

guidellm export "$SOURCE_JSON" \
  --output "kind=console" \
  --output "kind=html,path=${RESULTS_DIR}/report.html" \
  --output "kind=csv,path=${RESULTS_DIR}/report.csv" \
  --output "kind=yaml,path=${RESULTS_DIR}/report.yaml"
```

Use the outputs as follows:

- **Console:** immediate comparison tables in the terminal.
- **HTML:** the easiest report to browse and share.
- **CSV:** convenient for spreadsheets and custom analysis.
- **JSON:** canonical report containing configuration, aggregate metrics, and
  retained request details.
- **YAML:** a human-readable representation of the full report, but usually
  larger than CSV or HTML.

GuideLLM also supports `kind=plot`, but this installation does not currently
include its plotting dependency. Install GuideLLM with its `plot` extra before
using it, then add:

```bash
--output "kind=plot,path=${RESULTS_DIR}/report.png"
```

## Reading the important metrics

Focus on these values for each concurrency level:

- **Request latency:** complete time for one turn.
- **TTFT:** time to first token; heavily influenced by prompt processing and
  queueing.
- **ITL:** inter-token latency during streamed generation.
- **Output tokens/second:** aggregate server output throughput.
- **Request concurrency:** actual observed concurrency, which can be below the
  configured stream count near the end of a run.
- **Completed, incomplete, and errored requests:** verify these before trusting
  latency or throughput comparisons.

Median and p95 latency are generally more useful for user experience than the
mean alone. Compare throughput together with latency: higher concurrency may
increase aggregate throughput while making each agent turn much slower.

## Results from the completed run

All 120 turns completed, with no errors or incomplete requests:

| Users | Completed turns | Mean request latency | Mean TTFT | Mean output throughput |
|---:|---:|---:|---:|---:|
| 1 | 8 | 25.90 s | 7.25 s | 20.05 tokens/s |
| 2 | 16 | 40.94 s | 11.85 s | 25.23 tokens/s |
| 4 | 32 | 51.86 s | 15.03 s | 23.79 tokens/s |
| 8 | 64 | 102.91 s | 23.50 s | 29.21 tokens/s |

The eight-user point achieved the highest aggregate output throughput, but its
mean turn latency was approximately four times the one-user latency. The result
therefore shows a capacity/latency tradeoff rather than an unqualified speedup.

## Reproducibility checklist

Record the following alongside every report:

- GuideLLM version and scenario file.
- Model name and model artifact or revision.
- Inference-server version and startup command.
- GPU type/count and tensor/pipeline parallel settings.
- Prefix-caching setting and KV-cache configuration.
- Any warm-up performed before measurement.
- Output directory or run identifier.

The JSON report already captures the GuideLLM configuration and detected model,
but server startup flags and hardware details should be saved separately.
