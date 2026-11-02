# inference-headroom

A learning PoC in inference optimization. It serves DeepSeek-V4.1-Flash-1017
for a coding-agent workload on one 8×H200 node with vLLM, optimizes it against
latency and throughput targets without losing quality, and explains every gain
with a measurement.

- [Goal](docs/goal.md)
- [Getting started](docs/getting-started.md): reach the H200 node and serve the model with vLLM
- [PoC plan and done criteria](docs/poc.md): workload, targets, benchmarks, quality gates, experiments
- [Skills plan](docs/skills-plan.md): after the PoC, two shareable Claude skills, one for prefill optimization and one for decode
- [ADR 0002: Coding-agent PoC on 8×H200 with vLLM](docs/adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xh200-with-vllm.md)
- [ADR 0001: Qwen3.8-Flash-Next on 1×B300 with vLLM](docs/adr/0001-qwen3.8-flash-next-on-1xb300-with-vllm.md) (paused)
