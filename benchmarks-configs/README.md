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

