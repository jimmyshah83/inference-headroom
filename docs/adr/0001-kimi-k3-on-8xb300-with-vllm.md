# ADR 0001: Serve Kimi K3 on 8×B300 with vLLM, native MXFP4, comparing TP8 and TEP8

- **Status:** Proposed
- **Date:** 2026-10-14
- **Goal:** [docs/goal.md](../goal.md)

## Context

The project predicts and measures decode performance: first a dense model on
1×B300, then Kimi K3 on 8×B300 with speculative decoding. This ADR fixes the
serving stack, precision, parallelism layouts, and measurement protocol so later
results are comparable.

### Models (from each Hugging Face `config.json`)

| | Kimi K3 [1][2] | Qwen3.8-27B [3] |
|---|---|---|
| Parameters | 2.78T total, 104B active | ~27B dense |
| Layers | 93: 69 KDA (linear attention) + 24 gated MLA | 64: 48 Gated DeltaNet (linear) + 16 full attention (GQA, 4 KV heads) |
| FFN | 896 routed experts + 2 shared, top-16; expert FFN 3072; routed experts run in a 3584-dim latent (hidden 7168) | dense, 17408 |
| Checkpoint | Routed experts MXFP4 (QAT, group 32); attention, shared experts, dense layer, and `lm_head` in BF16; **1.561 TB** | BF16, plus an official FP8 checkpoint |
| Speculative decoding | No native MTP layer (`num_nextn_predict_layers: 0`); third-party drafts [6][7] | One built-in MTP layer |

Both models are **hybrid**: most layers keep a fixed-size recurrent state per
sequence, and a minority keep a KV cache that grows per token. The analytical
model must carry both terms from the start.

### Hardware (8×B300 HGX)

- Per GPU: 288 GB HBM3e, up to 8 TB/s [8]. NVIDIA's HGX page lists 2.1 TB total
  rather than 8×288 GB = 2.3 TB [9], so usable capacity must be **measured**.
- NVLink 5 through NVSwitch: 1.8 TB/s per GPU, 14.4 TB/s aggregate [9].
- Dense tensor throughput per GPU, derived from the per-node HGX figures [9]:
  FP4 ≈ 13.5 PFLOPS, FP8 ≈ 4.5 PFLOPS, BF16 ≈ 2.25 PFLOPS.

### Fit check for Kimi K3 (estimate, to be replaced by the analytical model)

- An FP8 copy (~2.8 TB) does not fit. The native MXFP4 checkpoint does.
- Fully sharded across 8 GPUs, that is ~195 GB of weights per GPU. This leaves
  roughly **75–93 GB per GPU** for KV cache, KDA state, activations, CUDA graphs,
  and any draft model.
- The BF16 non-expert weights total ~114 GB (57.2B params, from the HF API).
  Data-parallel attention replicates them on every rank: 114 + 1,447/8 ≈
  **295 GB per GPU under DEP8, which does not fit**. This matches vLLM's
  description of K3 as able to "barely fit" on a DGX B300 and its use of DEP16
  for decode [4].

### Engine status

vLLM has supported K3 since launch and publishes an 8×B300 TP8 command [4]. Its
recipe points to the `vllm/vllm-openai:kimi-k3` image (CUDA 13, vLLM 0.29.0+,
driver r580+) and names the MoE backends `flashinfer_trtllm` (TP > 1) and
`deep_gemm_mega_moe` (DEP) [5]. K3 support was pre-release at launch, so flags
may change between images.

### Book grounding (Kiely, *Inference Engineering*, 2026)

- §2.4 (pp. 61–67) frames the work as roofline / arithmetic-intensity analysis.
- §2.2.4 (p. 54) warns that under batched inference "you should expect almost
  all of the model parameters to be active," which is exactly what the
  *unique experts touched per batch* term must quantify.
- §5.4 (pp. 142–147) covers the TP, EP, and PP tradeoffs and inter-GPU
  communication overhead.
- §5.2 (p. 129): speculative decoding "only improves TPS/ITL, not TTFT."
- §4.5.2 (p. 114): change one variable at a time, and expect speculative
  decoding to work against large batch sizes.

## Decision

1. **Engine: vLLM only.** Pin one Docker image digest per phase. Record the
   image digest, vLLM commit, driver, CUDA, FlashInfer, and the model revision
   (K3 `f831ab6`) with every run.
2. **Precision: native checkpoints.**
   - K3: the MXFP4 checkpoint as shipped.
   - Dense: BF16 vs the official FP8 checkpoint.
   - KV cache: FP8 by default for K3 (vLLM's benchmark setting [4]), with BF16 as
     one comparison point.
3. **Parallelism on 8×B300: TP8 vs TEP8.**
   - **TP8:** attention and experts are both tensor-sharded, so each expert's
     FFN is split 8 ways.
   - **TEP8:** attention is TP8 and routed experts are expert-parallel, 112 per
     GPU (`--enable-expert-parallel`).
   - **Stretch:** DP2×TP4 attention + EP8 (~210 GB per GPU, estimated), to
     observe all-to-all dispatch under data-parallel attention.
   - **Excluded:** DEP8, because it does not fit.
4. **Calibrate on dense first.** Run Qwen3.8-27B on 1×B300 and keep it short.
   The same analytical model, harness, and profiling workflow carry over to K3
   unchanged.
5. **Measurement protocol.**
   - **Load:** `vllm bench serve` against the OpenAI-compatible server, with
     fixed input/output lengths (ignore EOS) and concurrency in
     {1, 2, 4, 8, 16, 32, 64, 128, 256}.
   - **Shapes:** ~10k in / 1.5k out (matching Artificial Analysis's default) and
     1k in / 1k out (decode-dominated).
   - **Metrics:**
     - TTFT and TPOT/ITL (p50 and p90).
     - Per-user and aggregate output tok/s.
     - Bandwidth utilization = modeled bytes per decode step ÷ measured step
       time ÷ peak HBM bandwidth.
   - **Cold vs warm:** cold is the first requests after server start; warm is
     after a warm-up pass. Prefix caching is off for the main sweep. Server
     start and weight-load time are recorded as their own measurement.
   - **Repeats:** at least 3 runs per point, reporting the mean and spread. Change
     one variable at a time.
   - **Profiling:** Nsight Systems plus the vLLM torch profiler on one decode
     config per model. Attribute time to GEMMs (expert vs other), attention
     (KDA/DeltaNet vs MLA/GQA), communication (all-reduce, all-to-all), and
     idle or launch gaps.
6. **Speculative decoding.**
   - **Dense:** the built-in MTP layer.
   - **K3:** `Inferact/Kimi-K3-DSpark` (~4B BF16, 7 draft tokens, block-diffusion
     drafter) [6] and `lightseekorg/kimi-k3-eagle3.1-mla` (EAGLE-3.1, 3 steps) [7].
     Both publishers measured on GB200/GB300 at TP16. Acceptance at TP8 on B300
     is ours to measure.
   - **Prompts:** use real text (e.g., MT-Bench, SPEED-Bench), not random tokens,
     because acceptance on random tokens is meaningless. Fix temperature and
     top_p, since acceptance depends on them.
   - **Sweep across concurrency.** Gains should shrink as the batch grows.
     LightSeek reports 2.15× at concurrency 1 falling to 1.51× at 16 [7].
   - **Model term:** expected tokens per step comes from acceptance length;
     step cost is the target verifying k+1 tokens plus the draft cost.
7. **Operations.** Tear VMs down after every session. Keep the 1.56 TB of K3
   weights on persistent storage close to the GPUs so they are not
   re-downloaded each session.

## Alternatives considered

- **SGLang:** has supported K3 since launch, with strong DP-attention and DeepEP
  paths, but fewer published K3 speculative-decoding setups. One engine keeps
  the comparison about the model, not the engines. Revisit if vLLM's K3 support
  blocks progress.
- **DeepSeek V4.1 Flash as the MoE target (the earlier plan):** smaller and
  architecturally simpler. Replaced by K3, the harder and more current target.
- **Re-quantized K3 (e.g., NVFP4 community variants):** changes the model under
  test, and isn't needed because native MXFP4 fits on 8×B300.
- **DEP8, multi-node DEP16, disaggregated prefill/decode:** DEP8 does not fit;
  the others are out of scope (single node only).

## Consequences

- **Good:**
  - One engine, one node, and native weights keep the residuals attributable to
    the model and the parallelism layout, not to setup differences.
  - Dense calibration surfaces methodology errors cheaply before K3 GPU time is
    spent.
- **Memory is tight:**
  - With ~75–93 GB per GPU of headroom, the long shape at concurrency 256 may
    not fit. The analytical model should predict maximum concurrency per
    config *before* the sweep.
  - Under TP, vLLM duplicates MLA's compressed KV on every rank. Decode context
    parallelism (`--decode-context-parallel-size 8` in the recipe [5]) is a lever
    to evaluate.
  - Draft models consume HBM (DSpark ≈ 8 GB of BF16 weights), so speculative
    runs need their own capacity check.
- **Moving target:** K3 kernels and flags are changing between images. Pinning
  digests is mandatory, and upgrading mid-phase invalidates comparisons.
- **Cost:** loading 1.56 TB of weights each session is slow and billable. Batch
  K3 experiments into few, well-planned sessions.

## Open questions (resolve in the first K3 session)

1. Usable HBM per GPU (`nvidia-smi`): 288 GB, or closer to ~270 GB?
2. Which collectives vLLM issues for TEP8 (all-reduce, reduce-scatter/all-gather,
   or all-to-all), and whether dispatch moves 7168-dim hidden or 3584-dim latent
   activations. This sets the communication term.
3. MoE GEMM path: MXFP4×MXFP4 or MXFP4 weights × MXFP8 activations (K3 was
   trained W4A8). This decides whether the compute roof is the FP4 or the FP8
   rate.
4. Maximum concurrency that fits per shape and per config.

## References

- Book: Philip Kiely, *Inference Engineering* (Baseten Books, 2026): §1.4
  (pp. 35–38), §2.2.4 (pp. 53–54), §2.4 (pp. 61–67), Ch. 3 (pp. 71–93), §4.5
  (pp. 112–115), §5.2 (pp. 129–135), §5.4 (pp. 142–147), §7.2.1 (p. 186),
  §7.4.2 (p. 201).
- [1] Kimi K3 model card: https://huggingface.co/moonshotai/Kimi-K3
- [2] Kimi K3 `config.json` and HF API metadata (revision `f831ab6`, 2026-09-02):
  https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
- [3] Qwen3.8-27B `config.json`: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- [4] vLLM blog, "Kimi K3 Is Here: Efficient Day-0 Support on vLLM" (2026-07-27):
  https://vllm.ai/blog/2026-07-27-k3
- [5] vLLM recipe for Kimi K3: https://recipes.vllm.ai/moonshotai/Kimi-K3
- [6] DSpark draft for K3: https://huggingface.co/Inferact/Kimi-K3-DSpark
- [7] LightSeek EAGLE-3.1 draft for K3: https://huggingface.co/lightseekorg/kimi-k3-eagle3.1-mla
- [8] NVIDIA, "Inside NVIDIA Blackwell Ultra":
  https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- [9] NVIDIA HGX platform specifications: https://www.nvidia.com/en-us/data-center/hgx/
