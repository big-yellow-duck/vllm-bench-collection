# tp8-baseline

- **Hardware**: gfx1201
- **Model**: Qwen/Qwen3.8-27B-FP8
- **Workload**: simple-agent-multiturn-prefix-cache

## Serving configuration

- **Tensor parallel (TP)**: 8
- **Kernel / optimization tags**: baseline

## Workload parameters

- **Target endpoint**: http://100.82.150.22:8000
- **Configured streams**: [1, 2, 4, 8]
- **Turns**: 8
- **Prompt tokens**: 2048
- **Output tokens**: 512
- **Seed**: 20260715

## Result files

- `agent-multiturn-benchmarks.json`
- `bench.log`
