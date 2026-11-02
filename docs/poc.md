# PoC: coding-agent serving on 8×H200

Serve DeepSeek-V4.1-Flash-1017 for a coding-agent workload on one 8×H200 node,
then optimize it against latency and throughput targets without losing
quality. Explain every gain with a measurement. The window runs from Mon
2026-11-02 to **Fri 2026-11-20**.

- Setup: [getting-started.md](getting-started.md)
- Decisions: [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xh200-with-vllm.md)
- Study plan: [learning-plan.md](learning-plan.md)

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

For the cost figures, use the public on-demand H200 price, $5.40 per GPU-hour
([nebius.com/prices](https://nebius.com/prices), 2026-11-02). Keep any deal
pricing out of this repo.

## Baseline

The baseline is the recipe's H200 configuration from
[getting-started step 10](getting-started.md#10-start-the-two-replicas):
2 replicas, each TP4 with Engram offload, prefix caching on, and speculative
decoding off. Every experiment is compared against this baseline.

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

Run them in this order. For each one, write a prediction from the analytical
model before you measure, then the result and the explanation. Log them in
`docs/experiments.md`.

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| O1 | Parallel layout | 2×TP4 → 1×TP8 → DEP8 (`--data-parallel-size 8 --enable-expert-parallel`) | Per-GPU throughput vs per-user speed. TP8 halves per-GPU weights but doubles all-reduce traffic. The recipe lists DEP only for Blackwell, so TP8 and DEP8 are **unverified on H200** |
| O2 | Engram placement | offload on → off (TP8 only, where HBM may allow it) | What host lookups cost per decode step, and how much KV cache they buy |
| O3 | NUMA pinning | none → each replica's container bound to its GPUs' socket (`--cpuset-cpus`, `--cpuset-mems`) | Step time and tail latency, given the per-step Engram reads from host memory |
| O4 | Speculative decoding | off → DSpark, 5 tokens, adaptive verification (`--speculative-config` from the recipe); then 1–5 tokens | Acceptance length on coding traffic, and speedup vs concurrency. Expect the gain to shrink as batches grow (Kiely §5.2.1) |
| O5 | Scheduler token budget | default → `--max-num-batched-tokens` 2k / 8k / 16k; `--max-num-seqs`; `--max-cudagraph-capture-size` ≥ batch × (1 + 5 drafts) | TTFT vs inter-token latency: prefill chunks interfering with decode (Kiely §5.5) |
| O6 | Prefix caching | on → off (B2 and B3 only) | How much of the agent's TTFT the cache is saving |
| O7 | KV cache dtype | auto → `--kv-cache-dtype fp8`, if the image accepts it for this model on Hopper | KV capacity vs accuracy (gate A2) |
| O8 | Disaggregation (stretch) | 2×TP4 aggregated → 1 prefill + 1 decode, TP4 each, KV over NIXL with `vllm-router` | TTFT and inter-token latency on B3. Verified only on GB200 so far; Kiely §5.5.2 says it pays off only for prefill-heavy, high-volume traffic |

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
  - The compute ceiling for the experts. If they run W4A16, use the BF16 rate
    (~989 TFLOPS dense per GPU), not FP8.

  Then compare with B1. The target is decode within about 25% at five or more
  concurrency points. That is a judgment call, as in ADR 0001.
- **Profile two configurations,** one decode-heavy (8k/1k at high concurrency)
  and one prefill-heavy (64k/2k at low concurrency). Use vLLM's torch profiler:
  check `vllm serve --help` in v0.32 for the profiler flag, then call
  `/start_profile` and `/stop_profile` around a short run. Add Nsight Systems if
  it's available. Attribute the time to:
  - MoE GEMMs
  - attention and the sparse indexer
  - TP all-reduce
  - Engram host lookups and copies
  - launch gaps and idle time

## Done when

- [ ] **Reproducible serving.** The image digest, model revision and launch
  commands are in the repo. Someone else can bring both replicas up from
  getting-started alone.
- [ ] **Baseline measured.** B1, B2 and B3 sweeps done, with 3 repeats and zero
  failed requests.
- [ ] **Quality gates on the baseline.** A1 passes, and the A2 resolve rate is
  recorded.
- [ ] **Experiments.** At least five of O1–O7 have a prediction, a result and
  an explanation in `docs/experiments.md`.
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
- [ ] **Write-up.** About two pages, with three charts: throughput per GPU vs
  tok/s per user (the Pareto curve), TTFT vs concurrency, and DSpark
  acceptance and speedup vs concurrency.
- [ ] **Node released** on or before Fri 2026-11-20.

**Optional, if time allows:** O8 disaggregation, and a comparison with the
same model on B300 at the same targets.

## Schedule

| Week | Dates | PoC work |
|---|---|---|
| 1 | Mon 11-02 – Sun 11-08 | Get node access. Bring up both replicas. Run A1 and the baseline A2. Run B1 with the analytical model's predictions written down first. Analyze the B3 traces |
| 2 | Mon 11-09 – Sun 11-15 | B2 and B3 baselines. Experiments O1–O6. Both profiles |
| 3 | Mon 11-16 – Fri 11-20 | O7 and the final configuration. A2 and A3 on the final configuration. Package, write up, release the node |

If the node arrives late, cut O7 and O8 before cutting a quality gate.

## Where results go

- **Raw output stays on the node** under `~/runs/<benchmark>-<config>/`:
  AIPerf artifacts, server logs, `env.txt`, and SWE-bench logs. Copy what you
  need before releasing the node.
- **Summaries go in this repo:** `docs/experiments.md` (the log),
  `results/*.csv` (one row per measured point), and the write-up.
- **Keep internal details out of the repo:** customer names, deal pricing, and
  anything quoted from internal channels.
