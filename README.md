# inference-headroom

A learning PoC in inference optimization. It serves DeepSeek-V4-Flash-0731 for
a coding-agent workload on one B300 GPU with vLLM, optimizes prefill and
decode one change at a time, and explains every gain with a measurement.

- [Goal](docs/goal.md)
- [Getting started](docs/getting-started.md): reach the B300 node and serve the model with vLLM
- [PoC plan and done criteria](docs/poc.md): workload, the baseline sweep, the loop, quality checks, knobs
- [Skills plan](docs/skills-plan.md): after the PoC, two shareable Claude skills, one for prefill optimization and one for decode
- [ADR 0003: DeepSeek-V4-Flash-0731 on 1×B300 with vLLM](docs/adr/0003-deepseek-v4-flash-0731-on-1xb300-with-vllm.md)
- [ADR 0002: Coding-agent PoC on 8×B200 with vLLM](docs/adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md) (superseded)
- [ADR 0001: Qwen3.8-Flash-Next on 1×B300 with vLLM](docs/adr/0001-qwen3.8-flash-next-on-1xb300-with-vllm.md) (paused)
