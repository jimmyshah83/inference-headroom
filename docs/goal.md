# Goal

Predict, measure, and explain LLM decode performance on one NVIDIA B300. Start
with a dense model, then serve **Qwen3.8-Flash-Next** on the same **1×B300**,
optimized for performance and including speculative decoding. Flash-Next is an
MoE with a 125B main model (6B active) plus a 51B n-gram embedding, and its
official FP8 weights come from Hugging Face.

The core artifact is a hand-written analytical model that predicts throughput
and latency. It is checked against measurements, and every material gap is
explained by a profile. This is a learning project: documentation first, then
code and results.

**Target date:** 2026-12-31. Key decisions: [ADR 0001](adr/0001-qwen3.8-flash-next-on-1xb300-with-vllm.md).

## Phases

| # | Phase | What | Kiely, *Inference Engineering* |
|---|---|---|---|
| 1 | Dense calibration (keep it short) | Qwen3.8-27B on 1×B300. Hand-write the model: weight bytes, KV/state bytes, FLOPs per token, HBM-bandwidth decode bound. Sweep concurrency 1–256 on two input/output shapes; BF16 vs FP8; warm vs cold. Profile one decode config. Measure MTP speculative decoding. | §2.4, Ch. 3, §1.4, §4.5, §7.2.1 |
| 2 | MoE on 1×B300 | Qwen3.8-Flash-Next, FP8. Extend the model: active vs total params, bytes per step (non-expert weights stay BF16), unique experts touched per batch, the 2,048-token attention budget of Qwen Sparse Attention (QSA), and n-gram embedding lookups. Run the same sweep with the n-gram table in HBM vs offloaded to host memory. Profile one decode config: expert GEMMs, attention (Gated DeltaNet vs QSA), n-gram lookups and host transfers, and idle time. | §2.2.4, §5.3.2 |
| 3 | Speculative decoding on Flash-Next | Measure the acceptance length of the built-in MTP layer (1–3 draft tokens) and its effect on per-request speed and total throughput across concurrency. Fold it into the model. | §5.2 |
| 4 | Write-up | One post covering dense, then MoE, then speculative decoding. | §7.4.2 |

## Done when

- **Dense:** the model is checked against at least five measured concurrency
  points, with the residuals explained. Target: decode throughput within about
  25%. That is a judgment call, not a standard.
- **MoE:** predicted vs measured for Qwen3.8-Flash-Next on 1×B300 across
  concurrency. The unique-experts-touched term is explained, and the n-gram
  table in HBM is compared with the table offloaded to host memory.
- **Profiles:** two with time attribution, one dense and one MoE.
- **Speculative decoding:** acceptance length and speedup vs concurrency, measured
  on both models and compared with the model's prediction.
- **Write-up:** drafted.

## Non-goals

- Multi-GPU or multi-node serving (so no tensor or expert parallelism),
  disaggregated prefill/decode, or production SLOs.
- Comparing inference engines (vLLM only).
- Re-quantizing Qwen3.8-Flash-Next (e.g., NVFP4 variants); the official FP8
  checkpoint is used as shipped.
- Image and video inputs; serving is text-only.
- Quality evaluation beyond a sanity check that outputs are coherent.
