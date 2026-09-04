# tp2-rdna4-hip-ar-warm

- **Hardware**: gfx1201
- **Model**: Qwen/Qwen3.8-27B-FP8
- **Workload**: rdna4-tp2-hip-ar-warm-agent-multiturn-prefix-cache-c1

## Serving configuration

- **Tensor parallel (TP)**: 2
- **Kernel / optimization tags**: rdna4, hip-ar, warm

## Workload parameters

- **Target endpoint**: http://192.168.50.137:8000
- **Configured streams**: [1]
- **Turns**: 8
- **Prompt tokens**: 2048
- **Output tokens**: 512
- **Seed**: 20260715

## Result files

- `agent-multiturn-benchmarks.json`
- `agent-multiturn-benchmarks.csv`
