# Skills plan: prefill and decode optimization

**Status:** planned. Work starts after the PoC ends on Fri 2026-11-20, and
nothing is built yet.

Turn the PoC's procedure into two shareable Claude skills. Given a served model
and a latency target, Claude can then plan, run, and explain an optimization
pass, either for prefill or for decode, the way this PoC does by hand.

## Why split by prefill and decode

The two phases have different bottlenecks, metrics, and levers (Kiely §2.4,
pp. 61–67; §5.5, p. 148):

| | Prefill | Decode |
|---|---|---|
| Bottleneck | Compute (FLOPs) | Memory bandwidth (bytes per step) |
| Headline metric | TTFT; prefill tokens/s per GPU | Inter-token latency; tokens/s per user; throughput per GPU at a latency target |
| Analytical model | 2 × active parameters × prompt tokens, plus attention, ÷ achievable FLOPS | Bytes read per step ÷ HBM bandwidth, until batch size makes it compute-bound |
| Main levers | Prefix caching, chunked-prefill token budget, TP size, KV offload, cache-aware routing, disaggregation | Batch size and concurrency, CUDA graph sizes, speculative decoding, KV cache dtype, weight precision, TP vs EP/DP, host-offloaded state and NUMA |
| Workload that exposes it | Long prompts, multi-turn agents with prefix reuse | Short prompts with long outputs; high concurrency |

Some levers touch both phases. The chunked-prefill budget trades TTFT against
inter-token latency, and disaggregation separates the two phases entirely. Each
skill notes these crossovers and points to the other skill.

## Three conditions for building them

1. **Shared protocol, bundled in both.** Skills are shared one folder at a
   time, so each has to work alone. Both include the same
   `references/protocol.md`, covering benchmarking rules, environment capture,
   quality gates and guardrails. A check in this repo keeps the two copies
   identical.
2. **Built from evidence.** The skills encode what the PoC measured: which
   experiments mattered, which commands worked, and which numbers misled.
   They are written from `docs/experiments.md`, not from first principles.
3. **Tested somewhere new.** A skill that only reproduces this PoC on this
   node hasn't been tested. Run it on a different model or GPU before sharing.

## Skill 1: `prefill-optimization`

**Description** (draft; the description decides when Claude loads the skill):
"Measure and reduce time to first token and prefill cost for a served LLM:
long prompts, prefix caching, chunked prefill, interference between prefill
and decode, and whether to disaggregate. Use when TTFT misses a target, prompts
are long or agentic, or prefill limits capacity."

**Inputs:**
- an endpoint URL, or a launch command
- the model ID and its `config.json`
- GPU type and count
- engine and version
- the workload's input-length distribution and prefix reuse
- a TTFT target

**Procedure:**

1. Record the environment.
2. Characterize the workload: input lengths and prefix-reuse rate (for
   example, `aiperf analyze-trace`).
3. Predict TTFT and prefill throughput. State which compute ceiling applies.
   On B200, for example, that depends on whether the MXFP4 experts run at the
   FP4, FP8 or BF16 rate.
4. Measure the baseline:
   - TTFT vs input length at concurrency 1
   - TTFT p50 and p90 under target load
   - prefill tokens/s per GPU
   - prefix-cache hit rate
5. Run the experiments one at a time, in this order:
   1. prefix caching
   2. the chunked-prefill token budget
   3. TP size
   4. KV offload to host memory
   5. cache-aware routing across replicas
   6. disaggregation, only if Kiely §5.5.2's criteria are met (p. 149)
6. Run the quality gates after any change that affects numerics.
7. Report the results with their conditions, then give a recommendation.

**Seeded by PoC experiments:** O6, O5 and O8, and the prefill side of O1.

## Skill 2: `decode-optimization`

**Description** (draft): "Measure and raise decode speed for a served LLM:
inter-token latency, tokens per second per user, and throughput per GPU at a
latency target. Covers batching and concurrency, KV cache capacity and dtype,
speculative decoding, CUDA graphs, weight precision, TP vs EP, and
host-offloaded state. Use when inter-token latency misses a target or
throughput per GPU is too low."

**Inputs:** the same as the prefill skill, with an inter-token latency or
tokens-per-second-per-user target instead of a TTFT target.

**Procedure:**

1. Record the environment.
2. Predict:
   - bytes per step at batch sizes 1, 8, 32 and 128, including unique experts
     touched for MoE models
   - the batch size where decode becomes compute-bound (Kiely §2.4.1, p. 62)
   - the expected tokens per step with speculative decoding
3. Measure the baseline: a concurrency sweep with short prompts, long outputs,
   and EOS ignored, then goodput at the target.
4. Run the experiments one at a time:
   - concurrency and `--max-num-seqs`
   - CUDA graph capture sizes
   - speculative decoding: the method, the draft length, and acceptance on
     real text. Never use synthetic acceptance.
   - KV cache dtype
   - parallel layout
   - host-offloaded state and NUMA pinning
5. Run the quality gates: check that speculative decoding is lossless, and run
   an eval slice after any precision change.
6. Report the results with their conditions, then give a recommendation.

**Seeded by PoC experiments:** O1, O9, O4, O2, O3 and O7, and the decode-side
effect of O5.

## Shared protocol (`references/protocol.md`, in both skills)

- **Pin and record** the image digest, model revision, flags, driver and
  clocks.
- **Predict before measuring.** Write the number down first.
- **Change one variable at a time.** Re-run the baseline whenever anything
  under it changes.
- **Warm up, then take 3 repeats.** Report the mean and spread, and report per
  replica, per GPU and per node.
- **Zero errors.** A failed request invalidates the point.
- **Set targets explicitly**, measure goodput against them, and use real data
  for speculative decoding.
- **Quality gates:** compare runs task by task, and state how large a
  regression the eval can detect.
- **Two modes:**
  - **Plan mode** needs no GPU: estimates and an experiment plan, from
    `config.json` and the hardware specs.
  - **Run mode** needs an endpoint: the full measure-and-tune loop.
- **Guardrails:**
  - Ask before starting or stopping servers, or using shared GPU capacity.
  - Never leave GPUs running idle.
  - Keep customer names, prices and internal endpoints out of outputs.
  - Give every number with its conditions: model, precision, GPUs, engine,
    input/output lengths, concurrency, and cache state.

## Bundled scripts (extracted from the PoC)

- `scripts/env_capture.sh`: records the environment (getting-started step 12)
- `scripts/sweep.sh`: an AIPerf concurrency sweep with goodput and per-point
  artifact directories
- `scripts/to_csv.py`: AIPerf exports converted to one CSV row per point
- `scripts/estimate.py`: reads `config.json` and GPU specs and prints weights
  per GPU, KV per token, bytes per decode step, prefill FLOPs, and the
  crossover batch size. It is the analytical model, packaged.

## Where they live and how to share them

- **Source:** `skills/prefill-optimization/` and `skills/decode-optimization/`
  in this repo. Each holds `SKILL.md`, `references/` and `scripts/`.
- **To use one yourself:** copy or symlink its folder into `~/.claude/skills/`.
  To use it in another project, copy it into that project's `.claude/skills/`.
- **To share widely:** package both as a Claude Code plugin.
- Because these are meant to be shared, keep internal data out of them
  entirely.

## Build steps (after 2026-11-20)

| When | Step |
|---|---|
| Mon 11-23 – Wed 11-25 | Draft both `SKILL.md` files and `protocol.md` from `docs/experiments.md` and `poc.md`. Extract the scripts |
| Thu 11-26 – Fri 11-27 | Test plan mode without GPUs. Use the `skill-creator` skill to run evals: the skills should trigger on prefill or decode requests and stay quiet on unrelated ones. Check plan-mode estimates for two new models against their published vLLM recipes |
| Next GPU capacity | Test run mode on a model or GPU not used in the PoC. Qwen3.8-27B on one GPU is cheap; B300 would test whether the skills carry across hardware. Compare the skill's results with hand-measured ones |
| After that | Share with the SA team and collect feedback |

## Done when

- [ ] Both skills trigger on the right requests and not on unrelated ones,
  according to the description evals.
- [ ] In plan mode, the memory fit and minimum GPU counts for two models not
  used in the PoC match their published recipes.
- [ ] In run mode, on a new model or GPU, each skill produces, without manual
  steering:
  - a baseline
  - at least three experiments with predictions
  - a quality gate
  - a recommendation with conditions

  It must do this without breaking a guardrail.
- [ ] A colleague installs one skill from its README and runs it successfully.
