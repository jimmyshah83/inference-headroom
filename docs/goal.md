# Goal

Learn inference optimization by doing it end to end on a current model. The
aim is to prepare for an applied AI role on the **Eigen team** (Eigen AI, now
part of Nebius).

From **Mon 2026-11-02 to Fri 2026-11-20**:

- Serve **DeepSeek-V4.1-Flash-1017** for a **coding-agent** workload on one
  **8×H200** node with vLLM.
- Optimize it against latency and throughput targets without losing quality.
- Explain every gain with a measurement.

The core artifacts:

- A hand-written analytical model that predicts throughput and latency,
  checked against measurements, with every material gap explained by a
  profile.
- A packaged, reproducible endpoint configuration with its measured targets
  and quality results.

This is a learning project: documentation first, then code and results.

**Key decisions:** [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xh200-with-vllm.md).
**Plan and done criteria:** [poc.md](poc.md).
**After the PoC:** [skills-plan.md](skills-plan.md).

## Phases

| # | Phase | What | Kiely, *Inference Engineering* |
|---|---|---|---|
| 1 | Bring-up and baseline (week 1) | Work out the memory plan from `config.json` before touching the GPUs. Bring up two TP4 replicas from the recipe. Run the smoke and tool-call tests and the baseline SWE-bench slice. Predict, then measure, the fixed-shape sweeps. Analyze the agent traces, and measure the multi-turn and agent-trace baselines | §1.4, §2.4, Ch. 3, §4.3.1, §4.5 |
| 2 | Prefill optimization (Mon 11-09 – Tue 11-10) | TTFT and prefill cost. Prefix caching, then the chunked-prefill token budget, recording each change's effect on decode as well. Take the prefill-heavy profile | §2.4, §5.3, §5.5 |
| 3 | Decode optimization (Wed 11-11 – Sun 11-15) | Inter-token latency, tok/s per user, and throughput per GPU at the targets. In order: parallel layout (2×TP4 vs TP8 vs DEP8), batch and CUDA graph sizes, DSpark speculative decoding, then Engram placement, NUMA pinning and KV cache dtype. Take the decode-heavy profile | §2.4, §5.1, §5.2, §5.4 |
| 4 | PD disaggregation (Mon 11-16 – Tue 11-17) | Turn the two TP4 GPU groups into one prefill engine and one decode engine (1P1D, KV over NVLink via NIXL). Compare with the aggregated 2×TP4 layout at both target profiles, then apply the decision rule | §5.5 |
| 5 | Quality, packaging, write-up (Wed 11-18 – Fri 11-20) | Choose the final configuration. Re-run the quality gates on it (SWE-bench paired comparison, DSpark lossless check). Package the endpoint. Write the results up and present them. Release the node | §5.1.3, §7.1, §7.2.1, §7.4.2 |
| 6 | Skills (after 2026-11-20) | Turn the PoC's procedure into two shareable Claude skills, `prefill-optimization` and `decode-optimization`. Test them on a model or GPU the PoC didn't use. See the [skills plan](skills-plan.md) | §2.4, §5.5 |

## Done when

See the checklist in [poc.md](poc.md#done-when). In short:

- The baseline and final configuration are both measured against the
  interactive and async targets, with zero failed requests.
- Prefill and decode optimization each have their experiments logged with a
  prediction, a result and an explanation.
- PD disaggregation (1P1D) is measured against the aggregated 2×TP4 layout,
  with a recorded decision, or a documented reason it couldn't run.
- The final configuration beats the baseline on one profile. The target is
  ≥ 1.3×, which is a judgment call.
- The final configuration passes every quality gate.
- The analytical model matches decode within about 25% at five or more
  concurrency points (also a judgment call), with gaps explained by profiles.
- The packaged endpoint and the write-up are done, and the node is released by
  2026-11-20.

## Non-goals

- Multi-node serving. One node only, though TP and EP within it are in scope.
- Comparing inference engines (vLLM only).
- Re-quantizing the model. The official MXFP4/MXFP8 checkpoint is used as
  shipped; NVFP4 variants need Blackwell.
- Image and video inputs; serving is text only.
- Production SLOs, multi-tenancy, autoscaling.
- A B300 comparison. It is an optional stretch, not a requirement.

## Paused

The earlier plan, which measured decode on 1×B300 (Qwen3.8-27B, then
Qwen3.8-Flash-Next), is paused. Its decisions stay in
[ADR 0001](adr/0001-qwen3.8-flash-next-on-1xb300-with-vllm.md), and the B300
guide is in git history.
