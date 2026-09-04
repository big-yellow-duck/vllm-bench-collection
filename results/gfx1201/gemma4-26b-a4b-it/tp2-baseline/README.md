# tp2-baseline

- **Hardware**: gfx1201
- **Model**: google/gemma-4-26B-A4B-it
- **Workload**: simple-agent-multiturn-prefix-cache

## Serving configuration

- **Tensor parallel (TP)**: 2
- **Kernel / optimization tags**: baseline

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
