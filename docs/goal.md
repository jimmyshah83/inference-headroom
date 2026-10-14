# Goal

Predict, measure, and explain LLM decode performance on NVIDIA B300. Start with a
dense model on one GPU, then serve **Kimi K3** (2.8T-parameter MoE, weights from
Hugging Face) on **8×B300**, optimized for performance and including speculative
decoding.

The core artifact is a hand-written analytical model that predicts throughput
and latency. It is checked against measurements, and every material gap is
explained by a profile. This is a learning project: documentation first, then
code and results.

**Target date:** 2026-12-31. Key decisions: [ADR 0001](adr/0001-kimi-k3-on-8xb300-with-vllm.md).

## Phases

| # | Phase | What | Kiely, *Inference Engineering* |
|---|---|---|---|
| 1 | Dense calibration (keep it short) | Qwen3.8-27B on 1×B300. Hand-write the model: weight bytes, KV/state bytes, FLOPs per token, HBM-bandwidth decode bound. Sweep concurrency 1–256 on two input/output shapes; BF16 vs FP8; warm vs cold. Profile one decode config. Measure MTP speculative decoding. | §2.4, Ch. 3, §1.4, §4.5, §7.2.1 |
| 2 | MoE on 8×B300 | Kimi K3. Extend the model: active vs total params, unique experts touched per batch, all-to-all bytes per token (EP), all-reduce bytes (TP). Same sweep, TP vs EP. Profile one expert-parallel decode config: expert GEMMs, attention, all-to-all, idle. | §2.2.4, §5.4 |
| 3 | Speculative decoding on K3 | Measure acceptance length with the published K3 drafts and its effect on per-request speed and total throughput across concurrency. Fold it into the model. | §5.2 |
| 4 | Write-up | One post covering dense, then MoE, then speculative decoding. | §7.4.2 |

## Done when

- **Dense:** the model is checked against at least five measured concurrency
  points, with the residuals explained. Target: decode throughput within about
  25%. That is a judgment call, not a standard.
- **MoE:** TP vs EP for Kimi K3 on 8×B300, predicted vs measured, with the
  communication (all-to-all / all-reduce) cost explained.
- **Profiles:** two with time attribution, one dense and one MoE.
- **Speculative decoding:** acceptance length and speedup vs concurrency, measured
  on both models and compared with the model's prediction.
- **Write-up:** drafted.

## Non-goals

- Multi-node serving, disaggregated prefill/decode, or production SLOs.
- Comparing inference engines (vLLM only).
- Re-quantizing Kimi K3; the native MXFP4 checkpoint is used as shipped.
- Quality evaluation beyond a sanity check that outputs are coherent.
