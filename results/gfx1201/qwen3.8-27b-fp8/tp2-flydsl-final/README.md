# tp2-flydsl-final

- **Hardware**: gfx1201
- **Model**: Qwen/Qwen3.8-27B-FP8
- **Workload**: rdna4-flydsl-final-agent-multiturn-prefix-cache-c1

## Serving configuration

- **Tensor parallel (TP)**: 2
- **Kernel / optimization tags**: flydsl, final

## Workload parameters

- **Target endpoint**: http://localhost:8000
- **Configured streams**: [1]
- **Turns**: 8
- **Prompt tokens**: 2048
- **Output tokens**: 512
- **Seed**: 20260715

## Result files

- `agent-multiturn-benchmarks.json`
- `agent-multiturn-benchmarks.csv`
