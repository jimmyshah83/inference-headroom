# ADR 0003: DeepSeek-V4-Flash-0731 on one B300 with vLLM

- **Status:** Accepted
- **Date:** 2026-11-08
- **Supersedes:** [ADR 0002](0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md)'s
  hardware (8×B200), model (DeepSeek-V4.1-Flash-1017) and two-replica layout.
  The coding-agent workload, the AIPerf trace sweep and the quality checks
  carry over.
- **Goal:** [docs/goal.md](../goal.md). **Plan:** [docs/poc.md](../poc.md)

## Context

- **Hardware.** The 8×B200 node is gone. One B300 GPU is available, on a VM
  reached from the Mac as `b300-usn1-1` ([getting started](../getting-started.md)).
- **Model.** DeepSeek-V4.1-Flash-1017's checkpoint is about 511 GB, so it
  can't fit on one GPU. DeepSeek-V4-Flash-0731, its predecessor, is 166.9 GB.
- **Purpose.** Unchanged: learn prefill and decode by measuring them. With one
  GPU, the plan gets simpler. There are no parallel layouts, no second replica
  and no prefill/decode disaggregation, so each run changes one knob on one
  server.

### Model facts

From the model card [1], `config.json` and the file list at revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` [2], and the vLLM recipe [3]:

- **Size:** 284B parameters in total, about 13B active per token (recipe).
- **Layers:** 43, hidden size 4096.
- **MoE:** 256 routed experts, 6 per token, expert FFN size 2048,
  `sqrtsoftplus` routing with normalized top-k weights.
- **Attention:** hybrid compressed sparse attention (CSA) plus heavily
  compressed attention (HCA). The recipe puts the KV cache at 10% of
  DeepSeek-V3.2's at 1M tokens.
- **Checkpoint:** routed experts in FP4 (`expert_dtype: fp4`), the other
  weights in FP8 (E4M3, 128×128 blocks, UE8M0 scales), some in BF16.
  **166.9 GB (155.4 GiB) in 48 shards.** The largest file is 3.7 GB.
- **Context:** 1,048,576 tokens.
- **Speculative decoding:** one MTP layer (`num_nextn_predict_layers: 1`) and
  a DSpark draft module (the `dspark_*` keys in `config.json`).
- **Prompt format:** no Jinja chat template. vLLM encodes prompts itself with
  `--tokenizer-mode deepseek_v4`. Reasoning effort is `low`, `high` or `max`.
  In vLLM v0.33.0, **thinking is on at `high` unless a request passes
  `"thinking": false`** [4].
- **Sampling:** DeepSeek recommends temperature 1.0, with top_p 0.95 for
  agentic use and 1.0 otherwise [1].

### Hardware (one B300)

From ADR 0001's sources [5][6]:

- 288 GB of HBM3e, up to 8 TB/s. NVIDIA's HGX page lists 2.1 TB for eight
  GPUs, not 8 × 288 GB = 2.3 TB, so measure the usable capacity.
- Dense tensor throughput, derived from the per-node HGX figures: FP4 ≈ 13.5
  PFLOPS, FP8 ≈ 4.5 PFLOPS, BF16 ≈ 2.25 PFLOPS.
- Host link: PCIe Gen6 ×16. Nothing in this plan is offloaded to host memory.

### Fit

The 166.9 GB checkpoint leaves roughly 95–120 GB of HBM for the KV cache,
activations and CUDA graphs. That's an estimate: it depends on the usable
capacity and on vLLM's `--gpu-memory-utilization` (0.9 by default). The
startup log reports the real KV cache size.

## Decision

1. **Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` at revision
   `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, the official checkpoint as
   shipped.
2. **Engine:** vLLM v0.33.0, the same pinned image as before
   (`vllm/vllm-openai@sha256:d825baf9…a84c6e`). The recipe needs at least
   0.25.0 for this checkpoint. Its own B300 command uses v0.28.0, so v0.28.0
   is the fallback if v0.33.0 can't start the model.
3. **Layout:** one server on the one GPU: tensor parallel 1, with expert
   parallel turned on because the MoE kernel below requires it [4].
4. **Baseline flags**, from the recipe's single-GPU GB300 entry [3], which is
   the same chip family as the B300:

   ```
   --trust-remote-code --tensor-parallel-size 1 --enable-expert-parallel
   --moe-backend deep_gemm_mega_moe --tokenizer-mode deepseek_v4
   --attention-config '{"indexer_kv_dtype":"mxfp4"}'
   --kv-cache-dtype fp8 --block-size 256
   --tool-call-parser deepseek_v4 --enable-auto-tool-choice
   --reasoning-parser deepseek_v4
   --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
   ```

   with `VLLM_USE_RUST_FRONTEND=1`. Every flag was checked against the
   v0.33.0 source [4]. **Speculative decoding is off in the baseline.**
   Turning it on is an experiment.
5. **Workload:** unchanged. AIPerf replays the SemiAnalysis coding-agent
   traces under InferenceX's AgentX scenario
   ([`scripts/sweep_traces.py`](../../scripts/sweep_traces.py)).
6. **Baseline method:** run the full sweep (concurrency 1–32) once, with no
   repeats. Choose one operating concurrency from it, and measure every later
   change at that concurrency ([PoC plan](../poc.md#the-loop)).

## Alternatives considered

- **DeepSeek-V4.1-Flash-1017 on one B300.** About 511 GB, so it doesn't fit.
- **NVIDIA's NVFP4 re-quantization** (`nvidia/DeepSeek-V4-Flash-NVFP4`).
  The recipe says it can't use the DeepGEMM MegaMoE kernel. The PoC uses
  DeepSeek's checkpoint as shipped.
- **The preview checkpoint** (`deepseek-ai/DeepSeek-V4-Flash`). 0731
  supersedes it and adds DSpark.

## Consequences

- **Gone from the plan:** parallel layouts, the second replica, Engram
  placement, NUMA pinning, PD disaggregation and image input. 0731 is text
  only, and disaggregation needs at least two GPUs.
- **Results from the 8×B200 runs don't compare.** The model and hardware both
  changed.
- **No failover.** One GPU is a single point of failure, which matters if the
  results are ever compared with a managed service.

## Open questions

1. Does v0.33.0 start the model on the B300 with `deep_gemm_mega_moe`? The
   kernel requires an SM100-family GPU, FP4 experts, `sqrtsoftplus` routing
   and normalized top-k, and this checkpoint and GPU meet all four [4].
   One report had 0731 start on vLLM 0.26.0 and fail to start on 0.27.0 on
   8×H100 (vLLM #52065) [3]. That's Hopper, but it shows versions can break
   this model.
2. Can the KV cache hold one full 1M-token request? If startup refuses, pass
   `--max-model-len 393216`. That covers the 256K traces and is the recipe's
   minimum for `max` effort.
3. Which attention backend and MoE kernel does vLLM actually pick? Record both
   from the startup log.
4. How much HBM is usable, and how large is the KV cache?
5. Does speculative decoding (DSpark or MTP) start together with MegaMoE?

## References

- [1] DeepSeek-V4-Flash-0731 model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- [2] `config.json`, `model.safetensors.index.json` and the file list at
  revision `7872f01b…`, read on 2026-11-08
- [3] vLLM recipe for DeepSeek-V4-Flash, read on 2026-11-08:
  https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash (append `.md` and a
  `?hardware=` query for one configuration as text)
- [4] vLLM v0.33.0 source: `vllm/config/kernel.py` (MoE backends),
  `vllm/models/deepseek_v4/nvidia/model.py` (MegaMoE requirements),
  `vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py` (SM100 family),
  `vllm/tokenizers/deepseek_v4.py` (thinking and effort),
  `vllm/config/vllm.py` (breakable CUDA graphs for `DeepseekV4ForCausalLM`)
- [5] NVIDIA, "Inside NVIDIA Blackwell Ultra":
  https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- [6] NVIDIA HGX platform specifications: https://www.nvidia.com/en-us/data-center/hgx/
