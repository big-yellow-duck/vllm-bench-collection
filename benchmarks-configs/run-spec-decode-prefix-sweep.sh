#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

base_url="${BASE_URL:-http://100.82.150.22:8000}"
model="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
result_dir="${RESULT_DIR:-${repo_root}/results/gfx1201/qwen3.8-27b-fp8/spec-decode-sharegpt-prefix-sweep}"
bench_bin="${VLLM_BENCH_BIN:-vllm-bench}"
reset_prefix_cache="${RESET_PREFIX_CACHE:-1}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
conversations_per_stream="${CONVERSATIONS_PER_STREAM:-1}"

if ! command -v "${bench_bin}" >/dev/null 2>&1; then
  echo "error: '${bench_bin}' was not found" >&2
  echo "build the maintained vllm-bench binary from vllm/rust:" >&2
  echo "  cargo build --release -p vllm-bench" >&2
  echo "then set VLLM_BENCH_BIN=/path/to/vllm-bench" >&2
  exit 127
fi

mkdir -p "${result_dir}"

if [[ ! "${conversations_per_stream}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: CONVERSATIONS_PER_STREAM must be a positive integer" >&2
  exit 2
fi

dataset_path="${DATASET_PATH:-${repo_root}/.benchmark-data/sharegpt-prefix-sweep-8turn-${conversations_per_stream}perstream.json}"
if [[ -z "${DATASET_PATH:-}" && ! -f "${dataset_path}" ]]; then
  if [[ ! -x "${python_bin}" ]]; then
    echo "error: '${python_bin}' was not found or is not executable" >&2
    echo "set PYTHON_BIN to a Python environment containing 'datasets'" >&2
    exit 127
  fi
  "${python_bin}" "${script_dir}/prepare-sharegpt-prefix-sweep.py" \
    --output "${dataset_path}" \
    --conversations "$((8 * conversations_per_stream))" \
    --turns 8 \
    --seed 20260715
fi

dataset_args=(--dataset-name sharegpt --dataset-path "${dataset_path}")

reset_args=()
if [[ "${reset_prefix_cache}" == "1" ]]; then
  reset_args+=(--reset-prefix-cache)
elif [[ "${reset_prefix_cache}" != "0" ]]; then
  echo "error: RESET_PREFIX_CACHE must be 0 or 1" >&2
  exit 2
fi

"${bench_bin}" \
  --backend openai-chat \
  --base-url "${base_url}" \
  --model "${model}" \
  "${dataset_args[@]}" \
  --no-oversample \
  --multi-turn \
  --multi-turn-min-turns 8 \
  --multi-turn-max-turns 8 \
  --sharegpt-output-len 512 \
  --temperature 0 \
  --seed 20260715 \
  --num-warmups 1 \
  --sweep-max-concurrency 1,2,4,8 \
  --sweep-num-prompts-factor "${conversations_per_stream}" \
  "${reset_args[@]}" \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --sweep-summary-percentiles 95,99 \
  --save-result \
  --save-detailed \
  --label spec-decode-sharegpt-prefix-sweep \
  --metadata workload=spec-decode-sharegpt-prefix-sweep \
  --result-dir "${result_dir}"
