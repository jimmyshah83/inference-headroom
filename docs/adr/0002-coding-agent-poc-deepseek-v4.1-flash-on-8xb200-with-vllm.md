# ADR 0002: Coding-agent PoC: DeepSeek-V4.1-Flash-1017 on one 8×B200 node with vLLM

- **Status:** Proposed
- **Date:** 2026-11-02, revised 2026-11-03. The hardware changed from 8×H200
  to 8×B200, the baseline took on the vLLM recipe's B200 settings, and the
  engine moved from vLLM v0.32.0 to v0.33.0.
- **Supersedes:** [ADR 0001](0001-qwen3.8-flash-next-on-1xb300-with-vllm.md)'s
  hardware (1×B300), model (Qwen3.8-Flash-Next) and single-GPU scope, for the
  PoC window of 2026-11-02 to 2026-11-20. ADR 0001's measurement protocol still
  applies, except where this ADR changes it.
- **Goal:** [docs/goal.md](../goal.md). **Plan:** [docs/poc.md](../poc.md)

## Context

- **Hardware.** B300 capacity is scarce. An 8×H200 node was approved first,
  on 2026-11-02 in the Production tf-us1 project, but the account couldn't
  create a VM there. On 2026-11-03, capacity approved a temporary allocation
  of 8 B200 GPUs in the **Architects us-central1** project instead. The node,
  `jimmyshah-b200-instance-1`, runs in that project's default network
  ([getting started](../getting-started.md)).
- **Use case.** It changes from generic decode measurement to a **coding
  agent**: multi-turn, tool-calling, long contexts with heavy prefix reuse.
- **Purpose.** The project is now a learning PoC for inference optimization:
  serve a current model against latency and throughput targets, keep quality,
  and explain every gain.

### Model choice

DeepSeek-V4.1-Flash-1017 was released 2026-10-17 under the MIT license. It was
the newest open model that both fits on one 8-GPU node and leads open models
on coding-agent benchmarks.

All benchmark scores below are vendor-reported, from the model card. They are
not independent results.

| Model | Released | DeepSWE v1.1 | Terminal-Bench 4.0 | Fits 8×B200? |
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
  scales, embedding and LM head in BF16. About 511 GB (476 GiB) in total. The
  recipe's minimum GPU memory is 614 GB.

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
- **Recipe caveat:** the recipe's hardware checks and measured numbers,
  including its B200 lanes, were taken on the September (0909) checkpoint. The
  1017 weights share its config and size but "have not been separately re-run."

### Hardware (one 8×B200 node)

- **GPUs:** 8 × NVIDIA B200 with HBM3e. `nvidia-smi` reports 183,359 MiB
  (~179 GiB) per GPU, ~1,432 GiB per node. NVIDIA lists 180 GB per GPU, and
  Kiely §3.2.3 (p. 81) gives 192 GB. Bandwidth is up to 8 TB/s per GPU. NVLink
  5 gives 14.4 TB/s across the node, 1.8 TB/s per GPU.
- **Compute per GPU, dense:** FP4 ~9 PFLOPS and FP8 ~4.5 PFLOPS. These are
  NVIDIA's DGX B200 system figures divided by 8; Kiely gives FP8 as ~5
  PFLOPS. BF16 is ~2.25 PFLOPS, assumed at half the FP8 rate, because NVIDIA
  doesn't list it. Blackwell has **FP4 tensor cores**, so the MXFP4 experts
  can run natively.
- **Host:** 160 vCPU on 2× Xeon Platinum 8580 and 1,792 GiB RAM, in 2 NUMA
  nodes. GPUs 0–3 are on node 0 (CPUs 0–79), and GPUs 4–7 are on node 1
  (CPUs 80–159). Nebius preset `gpu-b200-sxm` / `8gpu-160vcpu-1792gb`.

### Fit and precision on Blackwell

- **Recipe default on B200:** TP2. It has these settings:
  - Engram offload to pinned host DRAM, read through UVA, with transparent
    huge pages
  - the `FLASHINFER_MLA_SPARSE_DSV41` attention backend with an MXFP4 indexer
  - fp8 KV cache
  - at most 8 sequences and 4,096 batched tokens

  The recipe says TP2 targets the low-concurrency (c1) agentic setting. It
  also lists DEP8 for B200 nodes, and its AgentX TP preset runs TP4 at
  concurrency 1–128. It gives no memory measurements for B200. The only
  measured figures are for Hopper: 81.2 GiB of resident weights and 38.5 GiB
  of KV cache per GPU at TP4.
- **Reference B200 configuration (TP2).** It matches the recipe's B200 TP2
  settings, and it turns on DSpark, which the recipe leaves opt-in:

  ```bash
  export VLLM_ENGINE_READY_TIMEOUT_S=7200
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_USE_RUST_FRONTEND=1

  vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --tokenizer-mode deepseek_v41 \
    --tensor-parallel-size 2 \
    --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
    --kv-cache-dtype fp8 \
    --kernel-config '{"enable_flashinfer_autotune":true}' \
    --gpu-memory-utilization 0.97 \
    --engram-config '{"cpu_offload":true,"use_thp":true}' \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092]}' \
    --max-cudagraph-capture-size 4092 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
    --language-model-only
  ```

  The CUDA graph sizes are multiples of 6, one target token plus 5 drafts per
  sequence.
- **Fit at TP4 (an estimate, to be measured):** a TP4 replica has ~716 GiB of
  HBM. With the offload, the weights took ~325 GiB at TP4 on Hopper. So TP4
  leaves far more room for KV cache than TP2 does. It may even hold the
  Engram tables without the offload, at ~46 GiB more per GPU (experiment O2).
- **MXFP4 experts on Blackwell:** the experts can run on FP4 tensor cores.
  The kernel vLLM v0.33.0 picks sets their compute ceiling: FP4, FP8 or BF16
  activations. Record it from the startup log.

## Decision

1. **Engine: vLLM only,** v0.33.0 pinned by digest
   (`vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`).
   v0.33.0 was released 2026-11-02. It was chosen over v0.32.0 before any
   baseline ran, for four reasons:
   - It updates the 1017 reference encoding (#62199, #62200).
   - It fixes DeepSeek-V4.1 Engram and compressed-KV bugs.
   - Its Rust frontend rejects unsupported flags instead of ignoring them.
     (It still ignores `--enable-auto-tool-choice`, with a warning; see open
     question 4.)
   - It's still CUDA 13.0, which matches the node's driver.

   The vLLM recipe has B200 configurations for this model (TP2, DEP8) and
   requires v0.31.0 or later. vLLM also matches ADR 0001's engine choice. The
   recipe's AgentX presets use a nightly image; the PoC stays on the pinned
   release. The SGLang guide for B300 is retired.
2. **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
   `1bae0e1088993b9b82de08f59de5be5a34e42170`, text only
   (`--language-model-only`), with the `deepseek_v41` tool-call and reasoning
   parsers. Image input is a later step on one TP8 replica, after the text
   work ([poc.md](../poc.md#later-image-input-on-one-tp8-replica)).
3. **Baseline layout:** two TP4 replicas with the recipe's B200 settings from
   the reference configuration:
   - `FLASHINFER_MLA_SPARSE_DSV41` attention with an MXFP4 indexer
   - fp8 KV cache
   - FlashInfer autotuning
   - Engram offload with transparent huge pages
   - `VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_USE_RUST_FRONTEND=1`

   vLLM's defaults apply to everything else: memory utilization, scheduler
   budget and CUDA graph sizes. Compile mode stays at its default, `NONE`:
   v0.33.0 sets it back to `NONE` for this model even when `VLLM_COMPILE` is
   passed (open question 7). Speculative decoding is **off** in the baseline;
   DSpark is experiment O4.

   The recipe's TP2 default is not the baseline, for three reasons:
   - It targets concurrency ~1, at 8 sequences and 0.97 memory utilization,
     but both PoC targets need concurrency.
   - Two TP4 replicas put one replica on each NUMA node (O3).
   - Two TP4 replicas give the TP4 + TP4 split that 1P1D uses (O8).

   TP2 is a layout point in O1.
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
   - Release the node by Fri 2026-11-20, earlier if done, or sooner if
     capacity reclaims the temporary allocation.
   - Stop it between sessions.
   - Keep the checkpoint and vLLM's compile cache on persistent disk.
   - Keep customer names, deal pricing and internal-channel content out of
     the repo.

## Alternatives considered

- **8×H200 in Production tf-us1.** It was approved first, but the account
  couldn't create a VM in that project. The B200s were available instead.
- **Keep 1×B300 with Qwen3.8-Flash-Next (ADR 0001).** B300 time is scarce,
  and the model is not tuned for coding agents. The plan is paused, not
  abandoned.
- **The recipe's B200 default, four TP2 replicas.** Not the baseline (see
  decision 3). It is tested in O1.
- **MiniMax-M3.1-Flash.** It is smaller (252 GB) but scores lower on the
  vendor coding benchmarks above.
- **DeepSeek-V4.1-Flash-1017 and GLM-5.3 together.** Rejected for scope: in
  three weeks, one model studied in depth teaches more than two studied
  lightly.
- **SGLang.** No SGLang cookbook entry for this model was found on
  2026-11-02. It hasn't been rechecked for B200.
- **One TP8 replica, or DEP8.** The recipe lists DEP8 for B200 nodes; TP8 is
  unverified. Both are tested in O1 rather than adopted.
- **NVFP4 checkpoints, such as `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`.** They
  run on Blackwell, so they are now an option. They change the weights under
  test, though, so they would need their own quality gates. Not adopted; a
  possible stretch.

## Consequences

- **Multi-GPU is in scope.** TP all-reduce and, in O1, expert parallelism
  enter the analytical model. ADR 0001's single-GPU simplicity is gone.
- **Results are specific to B200.** Do not extrapolate them to H200 or B300.
  A B300 comparison at the same targets is an optional stretch.
- **Engram offload makes decode depend on host memory.** NUMA placement and
  PCIe traffic become measured variables (O2, O3).
- **PD disaggregation is in scope but unverified here.** The recipe verified
  1P1D only on GB200 NVL4, using a node per role on 8-GPU systems. One-node
  1P1D on B200 may fail to start. If it does, the failure is documented as the
  result.
- **First boots are slow.** The reference configuration sets a 7,200-second
  ready timeout, so sessions need planning.
- **The checkpoint is newer than the recipe's evidence.** Any startup or
  accuracy surprise may come from the 1017 weights.
- **The node is a temporary allocation.** Capacity may reclaim it before
  2026-11-20, so copy results off the node after each session.

## Open questions (resolve in week 1)

1. Which MoE kernel does vLLM v0.33.0 pick for the MXFP4 experts on B200
   (SM100)? Are the activations FP4, FP8 or BF16?
2. Actual KV pool per replica, and host RAM pinned per replica, for the Engram
   tables.
3. Does `FLASHINFER_MLA_SPARSE_DSV41` run at TP4? The recipe documents it with
   TP2. Do TP8 and DEP8 start?
4. Does v0.33.0 accept every setting in the reference configuration
   (`VLLM_USE_V2_MODEL_RUNNER`, `VLLM_USE_RUST_FRONTEND`, `use_thp`,
   `enable_flashinfer_autotune`)? The recipe's AgentX presets use a nightly
   image. v0.33.0's Rust frontend rejects unsupported `vllm serve` flags, so
   the first start answers this for flags. Unknown environment variables may
   still be ignored without an error. **Partly answered 2026-11-04:** every
   flag was accepted, but the frontend warned that `enable_auto_tool_choice`
   "currently has no effect in Rust frontend, ignoring". Getting-started
   step 11's tool-call request shows whether tool calls still come back as
   `tool_calls`.
5. What is DSpark's acceptance length on coding traffic at concurrency 1–32?
   Does adaptive verification help? The reference configuration turns it off.
6. Can a 1P1D layout (TP4 prefill and TP4 decode, KV over NIXL) run on a
   single node?
7. Does torch.compile (`--compilation-config '{"mode":"VLLM_COMPILE"}'`) start
   at TP4 with the V2 model runner? **Answered 2026-11-04: the flag is
   accepted, but the engine resolves compile mode to `NONE` anyway.** v0.33.0
   turns on breakable CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) by default
   for `DeepseekV41ForCausalLM`, and whenever they're on it sets compile mode
   to `NONE` (`vllm/config/vllm.py`). A source comment says this architecture
   "cannot torch.compile". So the reference configuration's `VLLM_COMPILE` has
   no effect either, and the baseline keeps the default. The norm_quant,
   act_quant and allreduce_rms fusions that the log lists as enabled are
   Inductor passes, so they most likely don't run. See getting-started's
   [CUDA graphs](../getting-started.md#cuda-graphs) section.

## References

- Philip Kiely, *Inference Engineering* (Baseten Books, 2026):
  - §2.4 (pp. 61–67)
  - §3.2.3 (p. 81)
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
- vLLM v0.33.0 release notes (published under the `v0.33.0rc3` tag): https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3
- vLLM v0.33.0 image tags: https://hub.docker.com/r/vllm/vllm-openai/tags
- NVIDIA DGX B200: https://www.nvidia.com/en-us/data-center/dgx-b200/
- AIPerf: https://github.com/ai-dynamo/aiperf
- mini-SWE-agent: https://mini-swe-agent.com/latest/
- Nebius VM types (B200): https://docs.nebius.com/compute/virtual-machines/types
- Nebius prices: https://nebius.com/prices
