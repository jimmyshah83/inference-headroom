# Goal

Learn **prefill and decode** in depth by optimizing a current model end to
end. The aim is to prepare for an applied AI role on the **Eigen team**
(Eigen AI, now part of Nebius).

From **Mon 2026-11-02 to Fri 2026-11-20**:

- Serve **DeepSeek-V4-Flash-0731** for a **coding-agent** workload on one
  **B300** GPU with vLLM.
- Measure a baseline across load, then pick one operating point.
- Put every change through one loop:
  1. A perf run on coding-agent traces.
  2. A quality run on coding tasks.
  3. A gap analysis that decides whether prefill or decode gets the next
     change.
- Explain every gain with a measurement.

The core artifacts:

- A hand-written analytical model that predicts latency, checked against
  measurements, with every material gap explained by a profile.
- A packaged, reproducible endpoint configuration with its measured results
  and quality checks.

This is a learning project: documentation first, then code and results. Keep
it simple: one GPU, one server, one change at a time.

**Key decisions:** [ADR 0003](adr/0003-deepseek-v4-flash-0731-on-1xb300-with-vllm.md).
**Plan and done criteria:** [poc.md](poc.md).
**After the PoC:** [skills-plan.md](skills-plan.md).

## Phases

| # | Phase | What | Kiely, *Inference Engineering* |
|---|---|---|---|
| 1 | Bring-up and baseline (Sun 11-08 – Mon 11-09) | Start the server from the recipe's single-GPU settings. Write the analytical model's predictions. Run the trace sweep once (concurrency 1–32), the prefill and decode probes, and the quality checks. Choose the operating concurrency from the sweep | §1.4, §2.4, §4.5 |
| 2 | Prefill and decode runs (Tue 11-10 – Tue 11-17) | One knob per run at the operating concurrency, chosen by the previous run's gap analysis. Prefill side: prefix caching and the chunked-prefill token budget. Decode side: speculative decoding (DSpark, MTP), batch and CUDA graph sizes, the MoE kernel, KV cache dtype and GPU memory use. Every run gets a perf run, a quality run and a log entry. Take a prefill-heavy and a decode-heavy profile | §2.4, §5.1–§5.4 |
| 3 | Final configuration and write-up (Wed 11-18 – Fri 11-20) | Put the final configuration through the loop, plus the paired SWE-bench comparison with the baseline and the speculative-decoding lossless check. Package the endpoint and write the results up | §5.1.3, §7.2.1 |
| 4 | Skills (after 2026-11-20) | Turn the PoC's procedure into two shareable Claude skills, `prefill-optimization` and `decode-optimization`. Test them on a model or GPU the PoC didn't use. See the [skills plan](skills-plan.md) | §2.4, §5.5 |

## Done when

See the checklist in [poc.md](poc.md#done-when). In short:

- The baseline sweep is complete with zero failed requests, and the operating
  concurrency is chosen with a written reason.
- Every run went through the loop, and each is logged with a prediction, perf
  and quality results, a gap analysis and a decision.
- The final configuration beats the baseline at the operating concurrency and
  passes every quality check.
- The analytical model matches decode within about 25% at five or more batch
  sizes (a judgment call), with gaps explained by profiles.
- The packaged endpoint and the write-up are done.

## Non-goals

- More than one GPU: tensor or expert parallelism across GPUs, extra
  replicas, prefill/decode disaggregation.
- Comparing inference engines (vLLM only).
- Re-quantizing the model. The official FP8/FP4 checkpoint is used as shipped.
- Image and video inputs. The model is text only.
- Production SLOs, multi-tenancy, autoscaling.

## Earlier plans

- **DeepSeek-V4.1-Flash-1017 on 8×B200** ([ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md)):
  superseded on 2026-11-08, when the node went away.
- **Qwen3.8 on 1×B300** ([ADR 0001](adr/0001-qwen3.8-flash-next-on-1xb300-with-vllm.md)):
  paused. Its measurement protocol still applies.
