# ADR 0002: Coding-agent PoC: DeepSeek-V4.1-Flash-1017 on one 8×H200 node with vLLM

- **Status:** Proposed
- **Date:** 2026-11-02
- **Supersedes:** [ADR 0001](0001-qwen3.8-flash-next-on-1xb300-with-vllm.md)'s
  hardware (1×B300), model (Qwen3.8-Flash-Next) and single-GPU scope, for the
  PoC window of 2026-11-02 to 2026-11-20. ADR 0001's measurement protocol still
  applies, except where this ADR changes it.
- **Goal:** [docs/goal.md](../goal.md). **Plan:** [docs/poc.md](../poc.md)

## Context

- **Hardware.** B300 capacity is scarce. One 8×H200 node was requested for
  Nov 2–20 and approved on 2026-11-02 in the Production tf-us1 project
  (`us-central1`). It runs in its own network, apart from the project's
  production resources ([getting started](../getting-started.md#prerequisites-a-network-and-subnet-for-the-node)).
- **Use case.** It changes from generic decode measurement to a **coding
  agent**: multi-turn, tool-calling, long contexts with heavy prefix reuse.
- **Purpose.** The project is now a learning PoC for inference optimization:
  serve a current model against latency and throughput targets, keep quality,
  and explain every gain.

### Model choice

DeepSeek-V4.1-Flash-1017 was released 2026-10-17 under the MIT license. It was
the newest open model that both fits on one 8×H200 node and leads open models
on coding-agent benchmarks.

All benchmark scores below are vendor-reported, from the model card. They are
not independent results.

| Model | Released | DeepSWE v1.1 | Terminal-Bench 4.0 | Fits 8×H200? |
|---|---|---|---|---|
| **DeepSeek-V4.1-Flash-1017** | 2026-10-17 | **77.0** | **42.3** | Yes (511 GB) |
| MiniMax-M3.1-Flash | 2026-10-17 | 73.7 | 38.1 | Yes (252 GB) |
| GLM-5.3 | 2026-08-25 | 66.9 | 37.9 | Yes (756 GB FP8) |
| Kimi K3 | 2026-06-13 | 67.5 | 12.6 | No (needs two nodes) |
| Ling-3.1-flash-1019 | 2026-10-27 | not reported | not reported | Yes (574 GB FP8) |

### Model facts

From the [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash),
updated 2026-10-25, and the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):

- **Parameters:**
  - 552B in the backbone.
  - 196.6B in the Engram n-gram tables (~183 GiB).
  - ~14B in the DSpark drafter.
  - Active per token: ~8B for prompt tokens, ~16B for output tokens.
- **Layers:** 40 at hidden size 5120. They split into a 20-layer encoder and a
  20-layer decoder. Prefill runs only the encoder; the decoder's global KV is
  projected from the encoder's final hidden state.
- **Attention:** a 128-token sliding window in every layer, plus compressed KV
  latents. An indexer keeps the best 512 latents per query. DeepSeek puts the
  global KV at **~890 bytes per token**.
- **MoE routing:** 6 of 384 routed experts per token, plus one shared expert.
- **Checkpoint:** routed experts in MXFP4, other weights in MXFP8 with UE8M0
  scales, embedding and LM head in BF16. About 511 GB (476 GiB) in total:

  | Component | Stored |
  |---|---|
  | Routed + DSpark experts (MXFP4) | 259.5 GiB |
  | Engram tables (FP8) | 183.1 GiB |
  | Attention, dense projections, routers (FP8) | 6.9 GiB |
  | Embedding, LM head, norms | 3.9 GiB |
  | Block scales | 21.9 GiB |

- **Context:** 1M tokens. Prompt encoding is in Python, via
  `--tokenizer-mode deepseek_v41`; there is no Jinja chat template.
- **Speculative decoding:** DSpark is the only method. V4.1 dropped MTP.
- **Recipe caveat:** the recipe's hardware checks and measured numbers were
  taken on the September (0909) checkpoint. The 1017 weights share its config
  and size but "have not been separately re-run."

### Hardware (one 8×H200 node)

- **GPUs:** 8 × 141 GB HBM3e (1,128 GB per node) at 4.8 TB/s each, with
  NVLink 900 GB/s between GPUs.
- **Compute per GPU, dense:** FP8 ~1,979 TFLOPS and BF16 ~989 TFLOPS
  (Kiely §3.2.1, p. 79, for FP8). There are **no FP4 tensor cores**.
- **Host:** 128 vCPU (Xeon Platinum 8468) and 1,600 GiB RAM. Nebius preset
  `gpu-h200-sxm` / `8gpu-128vcpu-1600gb`.

### Fit and precision on Hopper

- **Recipe default on H200:** TP4 with the Engram tables offloaded to pinned
  host RAM. The recipe reports 81.2 GiB of resident weights and 38.5 GiB of KV
  cache per GPU. One node therefore holds **two TP4 replicas**. The recipe's
  figure of 94 GiB offloaded per replica is smaller than the 183 GiB table
  size, so host memory use has to be measured.
- **MXFP4 experts on Hopper:** vLLM v0.32.0's release notes say MXFP4 W4A4
  checkpoints "fall back to W4A16 with a warning when no W4A4 kernel exists"
  and that Humming is preferred over Marlin on SM90. If the startup log
  confirms W4A16, expert weights are read as 4-bit but computed in BF16. The
  analytical model must then use the BF16 compute ceiling with 4-bit weight
  bytes.

## Decision

1. **Engine: vLLM only,** v0.32.0 pinned by digest
   (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`).
   This is the only engine with a published H200 configuration for this
   model, and it matches ADR 0001's engine choice. The SGLang guide for B300
   is retired.
2. **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
   `1bae0e1088993b9b82de08f59de5be5a34e42170`, text only
   (`--language-model-only`), with the `deepseek_v41` tool-call and reasoning
   parsers.
3. **Baseline layout:** the recipe's H200 default, two TP4 replicas with the
   Engram tables offloaded and vLLM's defaults for everything else.
   Speculative decoding is **off** in the baseline; DSpark is experiment O4.
4. **Optimization phases:**
   - **Prefill** (O6, O5).
   - **Decode** (O1, O9, O4, then O2, O3, O7).
   - **PD disaggregation (O8):** the two TP4 GPU groups become one prefill
     engine and one decode engine (1P1D, KV over NVLink via NIXL, behind
     `vllm-router`). It is compared with the aggregated layout on the same 8
     GPUs and adopted per target profile only if it passes the decision rule
     in [poc.md](../poc.md#pd-disaggregation).
5. **Workloads:**
   - synthetic fixed shapes (8k/1k, 64k/2k)
   - synthetic multi-turn conversations with a shared prefix
   - SemiAnalysis Claude Code traces, through AIPerf's
     `semianalysis_cc_traces_weka_062126_256k`
   - SPEED-Bench coding, for speculative-decoding acceptance
6. **Targets:**
   - **Interactive:** p90 TTFT ≤ 3 s on cache-hit turns and p90 inter-token
     latency ≤ 20 ms.
   - **Async:** p90 TTFT ≤ 30 s, maximizing throughput per GPU.

   Both are judgment calls, recorded in [poc.md](../poc.md).
7. **Quality gates:**
   - A1: a smoke test and 20 out of 20 well-formed tool calls.
   - A2: a fixed 50-task slice of SWE-bench Verified with mini-SWE-agent
     2.4.6, compared task by task.
   - A3: DSpark is lossless at temperature 0.

   An optimization that fails a gate is rejected.
8. **Measurement:** AIPerf 0.13 is the main load generator, with
   `vllm bench serve` as a cross-check. ADR 0001's rules still apply: pinned
   versions, warm-up, 3 repeats, and one variable at a time. Results are
   reported per replica, per GPU and per node.
9. **Operations:**
   - Release the node by Fri 2026-11-20, or earlier if done.
   - Stop it between sessions.
   - Keep the checkpoint and vLLM's compile cache on persistent disk.
   - Keep customer names, deal pricing and internal-channel content out of
     the repo.

## Alternatives considered

- **Keep 1×B300 with Qwen3.8-Flash-Next (ADR 0001).** B300 time is scarce,
  and the model is not tuned for coding agents. The plan is paused, not
  abandoned.
- **MiniMax-M3.1-Flash.** It is smaller (252 GB) but scores lower on the
  vendor coding benchmarks above.
- **DeepSeek-V4.1-Flash-1017 and GLM-5.3 together.** Rejected for scope: in
  three weeks, one model studied in depth teaches more than two studied
  lightly.
- **SGLang.** No H200 cookbook entry was found for this model.
- **One TP8 replica, or DEP8.** Both are unverified on H200; the recipe lists
  DEP only for Blackwell. They are tested as experiment O1 rather than adopted.
- **NVFP4 checkpoints, such as `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`.** They
  need Blackwell.

## Consequences

- **Multi-GPU is in scope.** TP all-reduce and, in O1, expert parallelism
  enter the analytical model. ADR 0001's single-GPU simplicity is gone.
- **Results are specific to H200.** Do not extrapolate them to B300. A B300
  comparison at the same targets is an optional stretch.
- **Engram offload makes decode depend on host memory.** NUMA placement and
  PCIe traffic become measured variables (O2, O3).
- **PD disaggregation is in scope but unverified here.** The recipe verified
  1P1D only on GB200 NVL4, using a node per role on 8-GPU systems. One-node
  1P1D on H200 may fail to start. If it does, the failure is documented as the
  result.
- **First boots are slow.** The recipe sets a 3,600-second ready timeout, so
  sessions need planning.
- **The checkpoint is newer than the recipe's evidence.** Any startup or
  accuracy surprise may come from the 1017 weights.

## Open questions (resolve in week 1)

1. Which MoE kernel does vLLM v0.32.0 pick for the MXFP4 experts on SM90, and
   is it W4A16?
2. Actual KV pool per replica, and host RAM pinned per replica, for the Engram
   tables.
3. Does TP8 start on H200? Does DEP8?
4. Does `--kv-cache-dtype fp8` work for this model's sparse attention on
   Hopper?
5. What is DSpark's acceptance length on coding traffic at concurrency 1–32?
6. Can a 1P1D layout (TP4 prefill and TP4 decode, KV over NIXL) run on a
   single node?

## References

- Philip Kiely, *Inference Engineering* (Baseten Books, 2026):
  - §2.4 (pp. 61–67)
  - §3.2.1 (p. 79)
  - §3.3.1 (p. 84)
  - §4.5 (pp. 112–115)
  - §5.1.3 (p. 128)
  - §5.2 (pp. 129–135)
  - §5.3 (pp. 136–141)
  - §5.4 (pp. 142–147)
  - §5.5 (pp. 148–150)
- DeepSeek-V4.1-Flash-1017 model card: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- vLLM recipe, DeepSeek-V4.1-Flash (updated 2026-10-25): https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
  (source: https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
- vLLM v0.32.0 release notes: https://github.com/vllm-project/vllm/releases/tag/v0.32.0
- AIPerf: https://github.com/ai-dynamo/aiperf
- mini-SWE-agent: https://mini-swe-agent.com/latest/
- Nebius VM types (H200): https://docs.nebius.com/compute/virtual-machines/types
- Nebius prices: https://nebius.com/prices
