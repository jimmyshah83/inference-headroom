# Learning plan: 3 weeks, ~20 hours a week

The goal is to build the skills an applied AI engineer on the **Eigen team**
(Eigen AI, now part of Nebius) uses every day, and to have evidence of them by
Fri 2026-11-20. The [PoC](poc.md) is the hands-on core. This plan adds the
study, drills and write-up around it.

## What the role does

The team takes a new open model and ships an optimized, Docker-packaged
endpoint. It has to meet a customer's latency and throughput targets and pass
accuracy benchmarks, and it is validated with load tests such as AIPerf. Often
this happens on the day the model is released. That breaks down into five
skills:

| Skill | What it looks like | Where the PoC exercises it |
|---|---|---|
| Model bring-up | Read `config.json` and the checkpoint, then plan memory, parallelism and engine flags before touching a GPU | Week 1 worksheet, getting-started |
| Benchmarking | Choose realistic workloads, set targets, measure goodput correctly, and avoid misleading numbers | B1–B4, the target table |
| Optimization | Quantization, speculative decoding, caching, parallelism, scheduling, disaggregation, each with a prediction and a measurement | O1–O8 |
| Profiling | Find where the time goes, down to the kernel level | Two profiles, analytical model |
| Communication | Turn the results into a clear claim a customer or seller can use | Write-up, packaged endpoint |

## How the 20 hours split each week

| Block | Hours | What |
|---|---|---|
| Hands-on | ~12 | PoC tasks for that week ([schedule](poc.md#schedule)) |
| Study | ~5 | Book sections, papers, docs, code reading, listed below |
| Write and share | ~3 | Experiment log, notes, and one short post or readout |

Kiely's *Inference Engineering* is cited by section and printed page.

## Week 1 (Mon 11-02 – Sun 11-08): bring-up and measurement

**Study (~5 h)**

- Kiely §1.4 "Measuring Latency and Throughput" (pp. 35–38): percentiles and
  end-to-end metrics.
- Kiely §2.4 "Calculating Inference Bottlenecks" (pp. 61–67): ops:byte ratio,
  arithmetic intensity, and why decode is memory-bound.
- Kiely §3.1–3.3 (pp. 74–86): GPU architecture, Hopper (§3.2.1, p. 79) and
  multi-GPU instances (§3.3.1, p. 84).
- Kiely §4.3.1 vLLM (p. 106) and §4.5 "Performance Benchmarking and Load
  Testing" (pp. 112–115).
- Read the [vLLM DeepSeek-V4.1-Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
  end to end. Its architecture, memory and tuning sections cover this model's
  unusual parts: the encoder-decoder split, two-tier sparse attention, the
  Engram tables and DSpark.
- Skim the architecture sections of the
  [DeepSeek-V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf).

**Drill: the bring-up worksheet.** Do this before the node is ready, then check
it against the vLLM startup log.

1. From `config.json` and the recipe's memory table: what are the resident
   weights per GPU at TP4 and at TP8, with and without the Engram offload?
2. How much KV cache does one 256k-token session use? The model uses about 890
   bytes per token of global KV plus a 128-token sliding window per layer.
   How many such sessions fit in one replica?
3. Estimate the bytes read per decode step at batch sizes 1, 8, 32 and 128.
   The experts are MXFP4, so count unique experts touched per batch (6 of 384
   per token, plus one shared expert). Then convert to the decode ceiling in
   tok/s per user at 4.8 TB/s per GPU.
4. Estimate prefill FLOPs for a 64k-token prompt, using about 8B active
   parameters per prompt token. If the experts run W4A16, the ceiling is BF16,
   ~989 TFLOPS dense per GPU.
5. Note where your estimates and the log disagree, and why.

**Write and share (~3 h):** the worksheet and baseline numbers, plus a short
note on what surprised you.

## Week 2 (Mon 11-09 – Sun 11-15): optimization techniques

**Study (~5 h)**

- Kiely §5.1 "Quantization" (pp. 120–129), especially §5.1.3 "Measuring
  Quality Impact" (p. 128).
- Kiely §5.2 "Speculative Decoding" (pp. 129–135). DSpark is a block drafter
  closer to EAGLE (§5.2.3, p. 133) than to draft-target decoding.
- Kiely §5.3 "Caching" (pp. 136–141): prefix caching, where to store the KV
  cache, cache-aware routing, and long context.
- Kiely §5.4 "Model Parallelism" (pp. 142–147): TP for latency (§5.4.1) and EP
  for throughput (§5.4.2).
- Kiely §5.5 "Disaggregation" (pp. 148–150).
- Papers:
  - [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
  - [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (chunked prefill and
    prefill/decode interference)
  - [DistServe](https://arxiv.org/abs/2401.09670) (goodput, which AIPerf's
    `--goodput` implements)
  - [EAGLE](https://arxiv.org/abs/2401.15077)
  - [FlashAttention-3](https://arxiv.org/abs/2407.08608) (Hopper kernels)

**Drill: read the code (~1–2 h of the study time).** Read vLLM at tag
`v0.32.0`. Each week, follow one path and write a one-paragraph summary:

- The scheduler and its token budget (`--max-num-batched-tokens`), so you
  understand O5.
- The KV cache manager and prefix caching, so you understand O6.
- Speculative decoding: the DSpark proposer and verification, so you
  understand O4.
- The fused MoE layer: which MXFP4 kernel SM90 gets, so you understand the
  W4A16 question.

These lived under `vllm/v1/core/` and `vllm/model_executor/layers/fused_moe/`
in 2025. Search the tree if they have moved.

**Write and share (~3 h):** the experiment log entries for O1–O6, each with
prediction, result and explanation.

## Week 3 (Mon 11-16 – Fri 11-20, plus the weekend for the write-up): quality, packaging, communication

**Study (~5 h)**

- Kiely §7.1 "Containerization" (p. 179), for packaging the endpoint.
- Kiely §7.2.1 "Concurrency and Batch Sizing" (p. 186), §7.4.2 "Cost
  Estimation" (p. 201) and §7.4.3 "Observability" (p. 203).
- How others benchmark:
  - [InferenceX](https://inferencex.semianalysis.com/): Pareto curves at fixed
    interactivity.
  - Artificial Analysis provider pages: how they measure output speed and
    TTFT. Compare your numbers with the providers listed for DeepSeek-V4.1-Flash.
- [SWE-bench Verified](https://www.swebench.com/) and the
  [mini-SWE-agent docs](https://mini-swe-agent.com/latest/): what a resolve
  rate does and doesn't tell you.

**Drill: day-0 bring-up on paper.** For two other recent releases,
`MiniMaxAI/MiniMax-M3.1-Flash` (~428B total, 23B active, 252 GB) and
`inclusionAI/Ling-3.1-flash-1019` (560B total, 25B active, 574 GB FP8), work
out from `config.json` alone:

- the minimum GPUs on H200 and on B300
- the parallel layout you'd start with
- the decode ceiling
- which engine flags you'd expect to need

Then check your answers against their vLLM or SGLang recipes.

**Write and share (~3 h):** the PoC write-up and a 10-minute readout. Share it
with the SA team, and ask someone on the Eigen team to review the method.

## Optional stretch (only if ahead of schedule)

- Write one Triton kernel from the official tutorials (fused softmax or
  matmul) and benchmark it on one H200 against PyTorch. It's about 2 hours,
  and it makes Kiely §4.1 (pp. 96–101) concrete.
- Run O8 (one prefill and one decode replica on the same node), and compare
  your result with Kiely §5.5.2's rule for when disaggregation pays off.

## Habits for the three weeks

- **Predict before you measure.** Write the number down first. A wrong
  prediction you can explain teaches more than a right one.
- **One variable at a time** (Kiely §4.5.2, p. 114). Re-run the baseline when
  anything under it changes.
- **Every claim needs its conditions:** model, precision, GPUs, engine version,
  input/output lengths, concurrency, and cache state.
- **Twice a week, skim vLLM's merged pull requests** for DeepSeek-V4.1 and
  Hopper. New releases move the baseline.

## Self-check questions

You should be able to answer these out loud by 11-20, using your own numbers:

1. Why is decode memory-bound on H200, and at what batch size does this model's
   decode stop being memory-bound? How does that move if the experts run W4A16?
2. Why does the recipe use two TP4 replicas instead of one TP8 replica on
   H200, and what did O1 show?
3. What does the Engram offload cost per decode step, and what does it buy?
4. How does DSpark's adaptive verification work, and when does speculative
   decoding make things worse?
5. How do you turn a latency target into a goodput number, and why does
   goodput beat raw throughput for comparing configurations?
6. How did you show that an optimization didn't hurt quality, and how small a
   regression could your eval detect?
7. When would you split prefill and decode onto separate GPUs for this
   workload, and when not?
8. What would you change to serve the same model on B300?

## After 11-20: turn the PoC into skills

Writing the procedure down as two Claude skills, one for prefill and one for
decode, tests whether you understood it. A skill has to state:

- when each technique applies
- how to predict its effect
- how to measure it
- what rule decides whether to keep it

That is the repeatable bring-up an optimization team needs. See the
[skills plan](skills-plan.md) for the design, the build steps (starting
Mon 11-23) and how to test them.

## What you'll have to show

- [ ] This repo: reproducible setup, PoC plan, ADR, experiment log, results CSVs
- [ ] The bring-up worksheet, checked against real logs
- [ ] Two profiles with time attribution
- [ ] The packaged final configuration with its measured targets
- [ ] The write-up and the readout slides or notes
- [ ] Two day-0 bring-up worksheets for other models
- [ ] After 11-20: the `prefill-optimization` and `decode-optimization` skills,
  tested on a model or GPU the PoC didn't use
