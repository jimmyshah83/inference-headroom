# PoC: coding-agent serving on 8×B200

Serve DeepSeek-V4.1-Flash-1017 for a coding-agent workload on one 8×B200 node,
then optimize it against latency and throughput targets without losing
quality. Explain every gain with a measurement. The window runs from Mon
2026-11-02 to **Fri 2026-11-20**.

- Setup: [getting-started.md](getting-started.md)
- Decisions: [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md)

The PoC copies how an optimization team ships a model endpoint. You start from a
baseline, change one variable at a time, and the result has to meet the latency
and throughput targets and pass accuracy checks. The deliverable is a packaged,
reproducible configuration.

## The workload: a coding agent

A coding agent (Claude Code, OpenHands, mini-SWE-agent) is not a chatbot. Its
traffic looks like this:

- **Many turns per task.** Each turn re-sends the whole growing conversation:
  system prompt, tool definitions, files read, command output.
- **Heavy prefix reuse.** Most of each prompt is a prefix the server has seen
  before. Prefix caching decides TTFT more than raw prefill speed does.
- **Long contexts.** Tasks often run to tens or hundreds of thousands of tokens.
- **Tool calls on every turn.** A tool call that fails to parse breaks the task,
  however fast the server is.
- **Bursts.** Sub-agents and parallel tool calls arrive together.

Measure this before you tune it. Run `aiperf analyze-trace` on the trace
dataset (B3 below) and record its input-length distribution and prefix-reuse
rate. Those numbers tell you which optimizations can matter.

## Targets

There is no customer SLA, so these are starting points and judgment calls.
Write down your reasoning if you change them.

| Profile | Who is waiting | Latency targets | What you maximize |
|---|---|---|---|
| **Interactive** | A developer watching the agent work | p90 TTFT ≤ 3 s on turns that hit the prefix cache; p90 inter-token latency ≤ 20 ms (≥ 50 tok/s per user) | Concurrency per replica that still meets both targets (goodput) |
| **Async** | Nobody: background agents, batch refactors | p90 TTFT ≤ 30 s; zero failed requests | Output tokens/s per GPU, and cost per completed SWE task |

AIPerf computes goodput (requests meeting every target) with
`--goodput "time_to_first_token:3000 inter_token_latency:20"`. Units are ms.

For the cost figures, use the public on-demand B200 price, $8.50 per GPU-hour
([nebius.com/prices](https://nebius.com/prices), 2026-11-03). Keep any deal
pricing out of this repo.

## Baseline

The baseline is defined in
[getting-started step 10](getting-started.md#10-start-the-two-replicas). It
runs 2 replicas, each TP4, with the vLLM recipe's B200 settings:
- fp8 KV cache
- FlashInfer sparse-MLA attention with an MXFP4 indexer
- Engram offload with transparent huge pages
- FlashInfer autotuning

Prefix caching is on, speculative decoding is off, and everything else uses
vLLM's defaults. The recipe's own B200 default is TP2 tuned for concurrency
~1. That configuration is the TP2 point in O1
([ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md),
decision 3). Every experiment is compared against this baseline.

## Performance benchmarks

Use [AIPerf](https://github.com/ai-dynamo/aiperf) 0.13 as the main load
generator. Use `vllm bench serve` as a cross-check. Run benchmarks from the
node, against one replica (4 GPUs) unless the benchmark says otherwise, so
network latency stays out of the numbers.

```bash
# Install AIPerf in its own virtual environment on the node.
python3 -m venv ~/aiperf && . ~/aiperf/bin/activate && pip install aiperf==0.13.0
```

**B1. Fixed shapes (roofline calibration).** Use two shapes, 8k in / 1k out
and 64k in / 2k out. EOS is ignored, so output lengths are exact. Thinking is
off.

```bash
# Sweep concurrency on replica A at 8k in / 1k out, with enough requests per
# point for stable p90s. Repeat each sweep 3 times.
for c in 1 2 4 8 16 32 64 128; do
  aiperf profile --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --url http://localhost:8000 --endpoint-type chat --streaming \
    --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --isl 8000 --osl 1000 \
    --extra-inputs '{"ignore_eos": true, "chat_template_kwargs": {"thinking": false}}' \
    --concurrency $c --request-count $(( c < 8 ? 64 : 8 * c )) \
    --warmup-request-count $c \
    --goodput "time_to_first_token:3000 inter_token_latency:20" \
    --artifact-dir ~/runs/b1-8k1k-baseline/c$c
done
```

**B2. Multi-turn with a shared prefix (synthetic agent).** Each run has 8-turn
conversations, an 8k-token shared system prompt, and 24k tokens of per-session
context. This isolates prefix caching.

```bash
# 64 conversations of ~8 turns each, back to back, at concurrency 16.
aiperf profile --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --conversation-num 64 --conversation-turn-mean 8 \
  --shared-system-prompt-length 8000 \
  --user-context-prompt-length 24000 --num-dataset-entries 64 \
  --isl 1500 --osl 500 \
  --extra-inputs '{"chat_template_kwargs": {"thinking": false}}' \
  --concurrency 16 --artifact-dir ~/runs/b2-multiturn-baseline
```

**B3. Real coding-agent traces.** These are SemiAnalysis's Claude Code traces,
the same dataset InferenceX used for its DeepSeek-V4.1-Flash agentic sweep.
Concurrencies are 1, 2, 4, 8, 16 and 32, InferenceX's task concurrencies.

```bash
# Replay the 256k-capped trace set at concurrency $c, without the recorded
# think-time gaps, after an agentic cache warm-up.
aiperf profile --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --ignore-trace-delays --agentic-cache-warmup-duration 120 \
  --concurrency $c --benchmark-duration 900 \
  --goodput "time_to_first_token:3000 inter_token_latency:20" \
  --artifact-dir ~/runs/b3-traces-baseline/c$c
```

Check `aiperf profile --help` for how this loader treats sessions and
timestamps before trusting the first result.

**B4. Speculative decoding acceptance.** Turn on DSpark (experiment O4) and run
the coding split of SPEED-Bench (`--public-dataset speed_bench_coding`). Read
acceptance from the server:

```bash
# Draft and accepted-token counters. Mean acceptance length is
# 1 + accepted_tokens / drafts.
curl -s localhost:8000/metrics | grep spec_decode
```

Never benchmark with synthetic acceptance. InferenceX's throughput mode fixes
the acceptance length (3.51) so runs are comparable, but those numbers don't
reflect real drafts. Use real rejection (`"rejection_sample_method":"block"`).

**Protocol** (carried over from ADR 0001):

- Pin the image, model revision, and flags. Record `~/runs/env.txt` for every
  session.
- Warm up before measuring, and repeat each point 3 times. Report the mean and
  spread.
- Change one variable at a time. Re-run the baseline whenever the image or
  driver changes.
- Report per replica, per GPU (divide by 4), and per node (both replicas).
- Report failed requests. Any error invalidates the point.

## Quality gates

An optimization that breaks the agent is not an optimization. Every
configuration you call "final" has to pass all three gates.

**A1. Smoke tests.** The answer to 17×19 must be 323. Then send 20 tool-call
requests (5 prompts × 4 repeats at the model card's temperature 1.0): all 20
must return a well-formed `tool_calls` entry
([getting-started step 11](getting-started.md#11-test-it-from-the-mac)).

**A2. SWE-bench Verified, a fixed 50-task slice, with mini-SWE-agent 2.4.6.**
Run it on the baseline and on the final configuration, using the same tasks,
sampling settings and reasoning effort.

```bash
# Install the agent and the SWE-bench harness.
python3 -m venv ~/swe && . ~/swe/bin/activate && pip install mini-swe-agent==2.4.6 swebench==5.0.2

# Tell LiteLLM the local model's context size and zero cost, so cost tracking
# doesn't fail on an unknown model.
cat > ~/swe/registry.json <<'EOF'
{"hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash-1017": {
  "max_input_tokens": 1048576, "max_output_tokens": 262144,
  "input_cost_per_token": 0, "output_cost_per_token": 0,
  "litellm_provider": "hosted_vllm", "mode": "chat"}}
EOF

# A model-only config layered over the default swebench.yaml. Passing -c
# replaces the default config, so list swebench.yaml first.
cat > ~/swe/dsv41.yaml <<'EOF'
model:
  model_name: "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash-1017"
  model_kwargs:
    api_base: "http://localhost:8000/v1"
    temperature: 1.0
    top_p: 1.0
    extra_body:
      chat_template_kwargs: {thinking: true, reasoning_effort: 50}
EOF

# Run tasks 0-49 of SWE-bench Verified with 8 agents in parallel.
LITELLM_MODEL_REGISTRY_PATH=~/swe/registry.json mini-extra swebench \
  -c swebench.yaml -c ~/swe/dsv41.yaml \
  --subset verified --split test --slice 0:50 --workers 8 \
  --output ~/runs/a2-swe-baseline

# Score the patches locally. The harness runs each task's tests in Docker.
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path ~/runs/a2-swe-baseline/preds.json \
  --max_workers 16 --run_id a2-baseline
```

- **Pass rule:** the final configuration must not resolve meaningfully fewer
  tasks than the baseline. With 50 tasks, the 95% confidence interval on a
  resolve rate is about ±13 points. Compare the two runs task by task (a paired
  comparison, McNemar's test), not as two headline numbers. This is the "small
  eval, big noise" problem from Kiely §5.1.3 (p. 128).
- **Sampling:** the model card says `temperature=1.0, top_p=1.0`, but the vLLM
  recipe says DeepSeek's published numbers used `top_p=0.95`. Pick one, record
  it, and keep it fixed.
- **Bonus load test:** the SWE run is a real agentic load. Scrape
  `localhost:8000/metrics` during it, and record wall time and GPU-hours per
  resolved task. That gives you the async profile's cost per task.
- **Check the file format:** mini-SWE-agent writes `preds.json`. If the harness
  rejects it, convert it to JSONL first.

**A3. Speculative decoding is lossless.** Send 50 prompts at temperature 0
with DSpark off, then with DSpark on, and compare the outputs. They should be
identical or very nearly so. Investigate every divergence. It is either
batch-dependent numerics, which is expected and worth understanding, or a bug.

## Optimization experiments

There are three optimization phases, in this order:

1. **Prefill:** TTFT and prefill cost.
2. **Decode:** inter-token latency, tokens/s per user, and throughput per GPU.
3. **PD disaggregation:** splits the node's two GPU groups into a prefill
   engine and a decode engine, and builds on the winners of the first two
   phases.

Within each phase, run the experiments in the order listed. For each one, write
a prediction from the analytical model before you measure, then the result and
the explanation. Each phase also feeds one of the future skills
([skills plan](skills-plan.md)).

### Prefill optimization

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| O6 | Prefix caching | on → off (B2 and B3 only) | How much of the agent's TTFT the cache is saving. Do this first, because it sets how much real prefill is left to optimize |
| O5 | Chunked-prefill token budget | default → `--max-num-batched-tokens` 2k / 8k / 16k | TTFT vs inter-token latency: large chunks finish prefills sooner but stall the decodes sharing the batch (Kiely §5.5). Record both metrics, because this knob affects decode too |

### Decode optimization

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| O1 | Parallel layout | 2×TP4 → 4×TP2 → 1×TP8 → DEP8 (`--data-parallel-size 8 --enable-expert-parallel`) | Per-GPU throughput vs per-user speed. TP2 is the recipe's B200 default; run it with the baseline's other settings, then once as the full reference configuration in ADR 0002 (DSpark on, 8 sequences). TP8 halves per-GPU weights vs TP4 but doubles all-reduce traffic. The recipe lists DEP8 for B200 nodes; TP8 is **unverified**. The layout also changes prefill speed; record TTFT too |
| O9 | Batch and CUDA graph sizes | default → `--max-num-seqs` 32 / 64 / 128; `--max-cudagraph-capture-size` ≥ batch × (1 + 5 drafts) | Where per-user speed falls off as the batch grows, and whether decode steps fall outside the captured graphs |
| O4 | Speculative decoding | off → DSpark with the reference configuration's `--speculative-config` (5 tokens, probabilistic drafts, block rejection, adaptive verification off); then adaptive verification on; then 1–5 tokens | Acceptance length on coding traffic, and speedup vs concurrency. Expect the gain to shrink as batches grow (Kiely §5.2.1) |
| O2 | Engram placement | offload on → off (at TP4 the tables add ~46 GiB per GPU, which a B200 may hold) | What host lookups cost per decode step, and how much KV cache they buy |
| O3 | NUMA pinning | none → each replica's container bound to its GPUs' socket: replica A `--cpuset-cpus 0-79 --cpuset-mems 0`, replica B `--cpuset-cpus 80-159 --cpuset-mems 1` | Step time and tail latency, given the per-step Engram reads from host memory |
| O7 | KV cache dtype | fp8 (baseline, from the recipe) → auto, if the attention backend accepts it | What fp8 KV costs in accuracy (gate A2) and buys in KV capacity |

### PD disaggregation

**O8. Compare two layouts on the same 8 GPUs:**

- **Aggregated (baseline):** two TP4 replicas, each doing both prefill and
  decode.
- **1P1D:** one TP4 prefill engine on GPUs 0–3 and one TP4 decode engine on
  GPUs 4–7. The prefill engine sends each request's KV cache to the decode
  engine over NVLink through vLLM's NIXL connector, behind
  `vllm-router --vllm-pd-disaggregation`.

Use the [recipe's PD commands](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
with these settings:

- the node's two GPU groups in place of its two GB200 trays
- a separate `VLLM_NIXL_SIDE_CHANNEL_PORT` for each engine
- each engine's flags carried over from the winners of the prefill and decode
  phases
- if DSpark won O4, DSpark in **both** engines, so the KV cache stays
  compatible

The recipe verified 1P1D only on GB200 NVL4. On 8-GPU nodes it uses a whole
node per role, so **one-node 1P1D on B200 is unverified.** That is also open
question 6 in ADR 0002.

- **Predict first:**
  - **KV transfer cost.** This model's KV is small: about 890 bytes per token
    by DeepSeek's figure, or about 2 KB by vLLM's measured pool. A 100k-token
    prompt is then roughly 0.1–0.2 GB, which should take well under a
    millisecond over NVLink. If so, transfer is not the cost to worry about.
  - **Decode capacity.** The real cost is that 1P1D has one decode engine
    where the aggregated layout has two. It wins only if removing prefill
    interference from decode is worth more than the lost decode capacity.
  - **Effect of the cache.** Coding-agent turns are mostly prefix-cache hits,
    so effective prefill per turn may be small. Expect gains mainly in
    inter-token latency during bursts of cold, long prompts.
- **Measure:**
  - B3 at concurrency 1–32 and B1 at 64k/2k, against both target profiles.
  - TTFT and inter-token latency at p50, p90 and p99. Look at the tail
    inter-token latency while long prefills run.
  - KV transfer time.
  - Prefill-engine and decode-engine utilization. A large imbalance means 1:1
    is the wrong ratio for this traffic.
- **Decision rule:** adopt 1P1D for a profile only if it improves goodput or
  p99 inter-token latency there without cutting async throughput per GPU by
  more than 10%. That threshold is a judgment call. Check the result against
  Kiely §5.5.2's criteria (p. 149).
- **Conditional disaggregation:** check whether `vllm-router` can send
  short or cache-hit prefills straight to the decode engine (Kiely §5.5.1,
  p. 148). If it can't, record that as a limitation.
- **If 1P1D won't start within half a day,** record the failure mode (error,
  versions, flags) and move on. A documented negative result still counts.

Log each experiment in `docs/experiments.md` using the same fields every time.
The skills are written from these entries.

```
## O<n>: <variable> (phase: prefill | decode | pd)
- Conditions: image, model revision, layout, workload, concurrency, cache state
- Flags changed: <baseline> → <variant>
- Prediction (and how it was made): …
- Command: …
- Result (mean ± spread, 3 repeats): …
- Explanation (profile or metric that shows why): …
- Decision rule learned: "use X when …, avoid when …"
```

The **final configuration** combines whichever variants won, all measured
together. Combinations interact, so measure them; don't add up the individual
gains.

## Analytical model and profiles

- **Predict first.** Before B1, estimate:
  - Decode step time at each concurrency, from bytes read per step ÷ HBM
    bandwidth.
  - Prefill time, from FLOPs ÷ achievable compute. This model's prefill runs
    only the 20-layer encoder half, about 8B active parameters per token,
    against about 16B per token in decode.
  - The compute ceiling for the experts, from the kernel vLLM picked for the
    MXFP4 experts (getting-started step 10). Dense per GPU: ~9 PFLOPS at
    FP4, ~4.5 at FP8, ~2.25 at BF16 (ADR 0002, Hardware).

  Then compare with B1. The target is decode within about 25% at five or more
  concurrency points. That is a judgment call, as in ADR 0001.
- **Profile two configurations,** one decode-heavy (8k/1k at high concurrency)
  and one prefill-heavy (64k/2k at low concurrency). Use vLLM's torch profiler:
  check `vllm serve --help` in v0.33 for the profiler flag, then call
  `/start_profile` and `/stop_profile` around a short run. Add Nsight Systems if
  it's available. Attribute the time to:
  - MoE GEMMs
  - attention and the sparse indexer
  - TP all-reduce
  - Engram host lookups and copies
  - launch gaps and idle time

  For the 1P1D layout (O8), also attribute the KV transfer between the
  engines.

## Done when

- [ ] **Reproducible serving.** The image digest, model revision and launch
  commands are in the repo. Someone else can bring both replicas up from
  getting-started alone.
- [ ] **Baseline measured.** B1, B2 and B3 sweeps done, with 3 repeats and zero
  failed requests.
- [ ] **Quality gates on the baseline.** A1 passes, and the A2 resolve rate is
  recorded.
- [ ] **Prefill optimization.** O6 and O5 each have a prediction, a result and
  an explanation in `docs/experiments.md`, with their effect on TTFT and on
  inter-token latency.
- [ ] **Decode optimization.** O1, O9 and O4 are done, plus at least one of O2,
  O3 and O7, each with a prediction, a result and an explanation.
- [ ] **PD disaggregation.** O8's 1P1D layout is measured against the
  aggregated 2×TP4 layout at both target profiles, and the decision rule is
  applied. If 1P1D couldn't run, the failure is documented.
- [ ] **Final configuration.** It beats the baseline on interactive goodput
  (concurrency per replica meeting both targets) **or** on async throughput per
  GPU. Report both deltas. The target is ≥ 1.3× on at least one, which is a
  judgment call.
- [ ] **Quality gates on the final configuration.** A1 passes; A2 shows no
  regression in the paired comparison; A3 shows DSpark is lossless, or the
  divergences are explained.
- [ ] **Target table.** For both baseline and final: the maximum concurrency
  meeting the interactive targets, throughput per GPU under the async targets,
  cost per 1M output tokens, and cost per resolved SWE task.
- [ ] **Analytical model.** Predicted vs measured decode step time at five or
  more concurrency points, with every gap over 25% explained by a profile.
- [ ] **Two profiles** (decode-heavy and prefill-heavy) with time attribution.
- [ ] **Packaged endpoint.** The final configuration as a single launch script
  (or compose file) plus a README listing its targets, measured results, and
  known limits.
- [ ] **Write-up.** About two pages, with four charts:
  - throughput per GPU vs tok/s per user (the Pareto curve)
  - TTFT vs concurrency
  - DSpark acceptance and speedup vs concurrency
  - aggregated vs 1P1D: TTFT and p99 inter-token latency on B3
- [ ] **Node released** on or before Fri 2026-11-20.

**Optional, if time allows:** a comparison with the same model on B300 at the
same targets.

**After the PoC:** turn the experiment log into two shareable Claude skills,
one for prefill and one for decode. See the [skills plan](skills-plan.md).

## Schedule

| Week | Dates | PoC work |
|---|---|---|
| 1 | Mon 11-02 – Sun 11-08 | Get node access. Bring up both replicas. Run A1 and the baseline A2. Run B1 with the analytical model's predictions written down first. Analyze the B3 traces. Run the B2 and B3 baselines |
| 2 | Mon 11-09 – Tue 11-10 | **Prefill optimization:** O6, O5. Prefill-heavy profile |
| 2 | Wed 11-11 – Sun 11-15 | **Decode optimization:** O1, O9, O4, then O2, O3, O7. Decode-heavy profile |
| 3 | Mon 11-16 – Tue 11-17 | **PD disaggregation:** O8, 1P1D vs 2×TP4 aggregated |
| 3 | Wed 11-18 – Fri 11-20 | Final configuration. A2 and A3 on it. Package, write up, release the node |

If the node arrives late, cut O7, then O3, then O2, before cutting PD
disaggregation or a quality gate.

## Where results go

- **Raw output stays on the node** under `~/runs/<benchmark>-<config>/`:
  AIPerf artifacts, server logs, `env.txt`, and SWE-bench logs. Copy what you
  need before releasing the node.
- **Summaries go in this repo:** `docs/experiments.md` (the log),
  `results/*.csv` (one row per measured point), and the write-up.
- **Keep internal details out of the repo:** customer names, deal pricing, and
  anything quoted from internal channels.
