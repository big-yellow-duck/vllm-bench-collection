# tp2-rdna4-fp8blockscalemm-candidate

- **Hardware**: gfx1201
- **Model**: Qwen/Qwen3.8-27B-FP8
- **Workload**: simple-agent-multiturn-prefix-cache

## Serving configuration

- **Tensor parallel (TP)**: 2
- **Kernel / optimization tags**: rdna4, fp8blockscalemm, candidate

## Workload parameters

- **Target endpoint**: http://localhost:8000
- **Configured streams**: [1, 2, 4, 8]
- **Turns**: 8
- **Prompt tokens**: 2048
- **Output tokens**: 512
- **Seed**: 20260715

## Result files

- `agent-multiturn-benchmarks.json`
- `agent-multiturn-benchmarks.csv`
- `agent-multiturn-benchmarks.c2-retry.json`
- `agent-multiturn-benchmarks.c2-retry.csv`
- `agent-multiturn-benchmarks.c8-isolated.json`
- `agent-multiturn-benchmarks.c8-isolated.csv`
