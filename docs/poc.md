# PoC: learn prefill and decode on 8×B200

**The goal is to understand prefill and decode in depth by measuring them.**
The vehicle is a real endpoint: DeepSeek-V4.1-Flash-1017 on vLLM, serving a
coding-agent workload on one 8×B200 node.

Every change goes through the same loop:

1. Measure speed on coding-agent traffic.
2. Check that the agent still solves coding tasks.
3. Ask which phase, prefill or decode, is missing its target. The answer picks
   the next change.

The window runs from Mon 2026-11-02 to **Fri 2026-11-20**.

- Setup: [getting-started.md](getting-started.md)
- Decisions: [ADR 0002](adr/0002-coding-agent-poc-deepseek-v4.1-flash-on-8xb200-with-vllm.md)
- Log: `docs/experiments.md`, one entry per run

## Prefill and decode

Every request runs in two phases (Kiely §2.4.2, p. 63):

| | Prefill | Decode |
|---|---|---|
| What it does | Runs the whole prompt through the model in one pass and writes its KV cache | Generates one token per step. Each step reads the active weights and the KV cache |
| Bound by | Compute (FLOPs) | Memory bandwidth (bytes per step), until the batch is large |
| The user sees it as | Time to first token (TTFT) | Inter-token latency (ITL), or tokens/s per user |
| Ways to make it cheaper | Skip it (prefix caching), size the chunks, use more GPUs per prompt, give it its own engine | Batch more requests per step, read fewer bytes per step, make more tokens per step (speculative decoding), launch less (CUDA graphs) |

The two phases share the GPUs. When a prefill chunk joins a batch, every
decode in that step waits for it. That coupling is why some changes (chunk
size, PD disaggregation) trade TTFT against ITL.

A coding agent adds one more metric. A tool call is useful only once it is
complete, so the agent waits for the whole response, not just the first token
(Kiely §1.4, p. 36). Record request latency as well as TTFT and ITL.

## The loop

Run 0 is the baseline. Every later run changes **one knob** from the current
reference configuration. If the change helps and passes the quality checks,
it becomes the new reference.

```
r00-baseline ─> 1 Predict ─> 2 Start ─> 3 Perf ─> 4 Quality ─> 5 Record ─> 6 Gap analysis
                    ^                                                            │
                    └───────────── next knob, from the prefill or decode menu ───┘
```

1. **Predict.** Before you measure, write down which metric should move, in
   which direction, by about how much, and why. Use the
   [analytical model](#analytical-model-and-profiles) where you can.
2. **Start.** Restart the replica with the change. Record what its startup log
   reports: KV cache tokens, CUDA graph memory, and the kernels it picked
   ([getting-started step 10](getting-started.md#10-start-the-two-replicas)).
   A start takes about 42 minutes.
3. **Perf.** AIPerf replays coding-agent traces against the replica, then runs
   two short phase probes. See [3. Perf run](#3-perf-run).
4. **Quality.** Run the A1 smoke tests, then a 50-task SWE-bench slice. See
   [4. Quality run](#4-quality-run).
5. **Record.** Add rows to `results/perf.csv` and `results/quality.csv`, and an
   entry to `docs/experiments.md`. See [5. Record](#5-record).
6. **Gap analysis.** Compare the run with the targets, the prediction and the
   reference run. Decide which phase is missing its target, and pick the next
   knob from that phase's menu. See [6. Gap analysis](#6-gap-analysis).

**Naming.** Name each run `r<NN>-<knob>-<value>`, for example
`r01-o6-prefix-cache-off`. Raw output goes under `~/runs/<run>/` on the node.
Copy the trace sweep's into this repo's `runs/<run>/`, which git ignores.

**Rules** (carried over from ADR 0001):

- Pin the image, model revision and flags. Append to `~/runs/env.txt` every
  session ([getting-started step 12](getting-started.md#12-record-the-environment-then-stop)).
- Warm up before measuring.
- Change one variable at a time, in the server and in the benchmark (Kiely
  §4.5.2, p. 114). Re-run the baseline whenever the image or driver changes.
- Report per replica and per GPU (divide by the replica's GPU count).
- Report failed requests. Any error invalidates the point.

**Time per run, estimated:**

| Step | Time |
|---|---|
| Start | ~42 min |
| Perf: trace sweep | ~2 h |
| Perf: phase probes | ~20 min |
| Quality: A1 | ~5 min |
| Quality: A2 | Measure it on the baseline. Estimate: 1.5–3 h |

That is 4–6 hours of node time per run, mostly unattended. Run them back to
back, and keep the node running between runs. A stopped node costs a
42-minute restart, and A2 fits best overnight.

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

Before run 0, run `aiperf analyze-trace` on the trace dataset and record its
input-length distribution and prefix-reuse rate. It needs no server, so run it
on the Mac. Those numbers tell you which phase can matter.ß

## 3. Perf run

Run every benchmark against replica A (port 8000), and keep replica B idle.
The two replicas share the host's CPUs and memory, and every decode step reads
Engram rows from host memory. Load on B can therefore move A's numbers (O3
tests this).

Run AIPerf on the node, inside tmux, for every run. From the Mac, every request
goes through the SSH tunnel: TTFT includes the round trip, the tunnel can bunch
tokens and skew ITL, and GPU telemetry isn't available. On 2026-11-06 a sweep
from the Mac also got 77 `Server disconnected` errors in its first five points,
while vLLM's own counters showed one aborted request. tmux keeps a sweep
running when the SSH connection drops or the laptop sleeps, so the Mac doesn't
have to stay awake.

```bash
# On the Mac (one time): install the pinned AIPerf from pyproject.toml into the
# repo's .venv, for `aiperf analyze-trace` and --dry-run.
uv sync && . .venv/bin/activate

# On the node (one time; done on 2026-11-06): the venv module isn't installed
# by default.
sudo apt-get install -y python3.12-venv
python3 -m venv ~/aiperf && . ~/aiperf/bin/activate && pip install aiperf==0.13.0
# AIPerf uses its own Hugging Face cache on the node, so the trace dataset stays
# out of the model's cache. ~/.bashrc sets HF_HOME=~/hf in every new shell and
# tmux window, so export this again in each one.
export HF_HOME=~/hf-aiperf

# Name the run and create its folder. Every command below writes under it.
RUN=r00-baseline && mkdir -p ~/runs/$RUN
```

The commands below were checked against `aiperf profile --help` for 0.13.0.
`--help` lists dataset names with hyphens, but the parser accepts only the
underscore spelling.

### Coding-agent traces (the workload)

These are SemiAnalysis's Claude Code traces, the same dataset InferenceX used
for its DeepSeek-V4.1-Flash agentic sweep. They are replayed under InferenceX's
AgentX scenario (`--scenario inferencex-agentx-mvp`), AIPerf's locked method
for this dataset. The concurrencies are InferenceX's: 1, 2, 4, 8, 16 and 32.

**Every run uses exactly this workload.**
[`scripts/sweep_traces.py`](../scripts/sweep_traces.py) holds the AIPerf flags
as fixed constants. Only the run name and the server change between runs, so a
difference in the results comes from the server. If a constant has to change,
run the baseline again with the new value.

The script runs one AIPerf profile per concurrency (1, 2, 4, 8, 16, 32), because
the scenario refuses a sweep in one command. Each point primes its
trajectories' caches, warms up for 2 more minutes, then measures for 15
minutes. The grace period lets responses that started before the deadline
finish and count. A full sweep takes about 2 hours.

**1. Copy the script to the node.** The repo is private, so the node can't
clone it. Copy the script again whenever it changes.

```bash
# On the Mac, from the repo root.
scp scripts/sweep_traces.py b200-usc1:~/
```

**2. Start a tmux session on the node.**

```bash
ssh b200-usc1
tmux new -s sweep    # tmux runs on the node; the Mac doesn't have it

# Inside tmux. Check that HF_HOME prints /home/<you>/hf-aiperf.
. ~/aiperf/bin/activate && export HF_HOME=~/hf-aiperf && echo $HF_HOME
```

**3. Run the sweep inside tmux.** Pass `--runs-dir ~/runs`, so the traces sit
with the rest of the run's raw output. Without it, the script writes next to
itself, to `/home/runs`, which it can't create.

```bash
# Print the AIPerf command for each point without running it.
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs --dry-run

# Test one point first: concurrency 1 only, about 20 minutes. Then run the
# check in step 5.
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs --concurrency 1

# Run 0: three passes of the sweep, to measure the noise between runs, about
# 6 hours. The c1 point from the test is reused.
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs --repeats 3

# A later run: one pass.
python ~/sweep_traces.py r01-o6-prefix-cache-off --runs-dir ~/runs
```

**4. Detach and come back.** Ctrl-b, then d, detaches; the sweep keeps
running on the node.

```bash
ssh b200-usc1
tmux attach -t sweep                            # the live output
cat ~/runs/r00-baseline/traces/progress.tsv     # one line per finished point
```

**5. Check every point.** AIPerf exits 0 and writes its summary even when it
cancels a point, so the script counts a cancelled point as done and skips it on
a rerun. This lists the points that had errors or were cancelled, and should
print nothing:

```bash
grep -l -E "Error Summary|INCOMPLETE" ~/runs/r00-baseline/traces/rep*/c*/profile_export_console.txt
```

For each point it lists, read `logs/aiperf.log` in its folder, delete the
folder, and rerun the same command.

**6. Copy the results to the Mac** when the sweep has finished.

```bash
# On the Mac, from the repo root. Git ignores runs/.
mkdir -p runs/r00-baseline && scp -r b200-usc1:runs/r00-baseline/traces runs/r00-baseline/
```

The script:

- **Checks the replica first.** It calls `/health` on the URL (default
  `http://localhost:8000`, replica A) and stops if the replica is down. For
  O8, point `--url` at the router.
- **Resumes.** A point that already has its summary
  (`profile_export_aiperf.json`) is skipped, so rerunning the same command
  picks up an interrupted sweep. Points without a summary are retried, but a
  cancelled point has one (step 5).
- **Stops cleanly.** Ctrl-C stops the whole sweep, not just the current point.
- **Logs progress.** It appends one line per point (start, end, exit code) to
  `<runs-dir>/<run>/traces/progress.tsv`.
- **Writes to `--runs-dir`.** The default is the `runs/` folder next to the
  script's parent folder, which is the repo's `runs/` on the Mac. On the node,
  pass `--runs-dir ~/runs`.

What the scenario does (from AIPerf 0.13.0's `inferencex_agentx_mvp.py`):

- **Agentic replay.** Each of the `c` sessions starts its trace at a random
  point (0–100% of the way through), and its context is primed before
  measuring. Each point therefore measures a steady state, not every session
  starting cold at turn 0. Sub-agent fan-out (the Task tool) is replayed too.
- **Recorded think times.** The gaps between an agent's turns (tool runs) are
  kept, each capped at 10 s. The scenario forbids `--ignore-trace-delays`.
  `c` is the number of concurrent agent sessions, not of requests in flight,
  so read the real batch size from `vllm:num_requests_running` in the server
  metrics. The decode probe gives clean batch sizes.
- **Exact output lengths.** It adds `ignore_eos`, so each turn generates
  exactly its recorded output length. Otherwise the model could stop early,
  and a change that alters what the model writes, like fp8 → auto KV, would
  also change how long decode takes.
- **Locks the method.** It requires streaming, at least 900 s per point, and
  the `first_turn_prefix` cache-bust marker (which it doesn't set for you).
  It rejects conflicting flags unless `--unsafe-override` is passed, which
  marks the run invalid. Don't use it for recorded runs.

The other flags hold the rest of the workload fixed:

| Flag | Without it |
|---|---|
| `--random-seed 42` | The scenario picks a fresh random seed for each run, so the traces and their starting points would differ from run to run |
| `--cache-bust first_turn_prefix`, and one AIPerf run per concurrency | The prefix cache holds 26.75M tokens and keeps them between points. Without a marker, each point would replay traces the previous point already cached, so later points would look faster. The marker is built from an ID that AIPerf creates fresh for every run (`uuid4`), so separate runs don't share cached conversations. It goes before the first user turn, so the shared system prompt can still hit the cache, as it would in real use |
| `--tokenizer-revision` | AIPerf builds the prompts with the tokenizer. A change to the tokenizer on Hugging Face could change prompt lengths |

Because the method matches InferenceX's, these results can be compared with
InferenceX's published AgentX results for this model, with the hardware and
configuration noted.

On 2026-11-06, a short test of this scenario on the node got through the
dataset download and about 2½ minutes of warm-up without errors before it was
stopped. A sweep from the Mac later that day hit the tunnel errors described
above: every point had errors, and AIPerf cancelled c4, c8 and c16 during
warm-up. Those results were deleted, so no valid point has run yet.

- **Run 0 only:** `--repeats 3` writes `traces/rep1`, `rep2` and `rep3`, to
  measure the noise between runs. Later runs use one pass. If a later run
  differs from its reference by less than twice the baseline's spread, repeat
  it before you believe it (Kiely §4.5.2, p. 114).
- **What each point writes** to `~/runs/<run>/traces/rep<r>/c<c>/` on the node:
  - `profile_export_aiperf.csv` and `.json`: the summary (TTFT, ITL, request
    latency, throughput, goodput), written when the point finishes
  - `profile_export.jsonl`: one line per request
  - `server_metrics_export.csv` and `.json`: the replica's `/metrics`, scraped
    during the run (on by default). This is where the split between prefill
    and decode comes from (see below)
  - `logs/aiperf.log`: check it first when a point fails
- **If AIPerf refuses a concurrency** because the trace pool is too small, the
  error says so. `--allow-dataset-wrap` reuses traces, but reused traces also
  hit the prefix cache, so record it if you use it.

### Phase probes

The traces mix prefill and decode. Two short synthetic runs measure each phase
on its own, so you can see which phase a change moved and compare it with the
analytical model.

```bash
# Prefill probe: 64k-token prompts, one output token, one and four at a time.
# TTFT here is almost all prefill, so prefill throughput = 64,000 / TTFT.
aiperf profile --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --isl 64000 --osl 1 \
  --extra-inputs '{"ignore_eos": true, "chat_template_kwargs": {"thinking": false}}' \
  --concurrency 1,4 --benchmark-duration 120 --benchmark-grace-period 120 \
  --warmup-request-count 2 \
  --artifact-dir ~/runs/$RUN/probe-prefill

# Decode probe: 1k in, exactly 1k out, at five batch sizes. ITL here is the
# time of one decode step at that batch size, which the analytical model
# predicts.
aiperf profile --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --isl 1000 --osl 1000 \
  --extra-inputs '{"ignore_eos": true, "chat_template_kwargs": {"thinking": false}}' \
  --concurrency 1,4,16,64,128 --benchmark-duration 180 --benchmark-grace-period 120 \
  --warmup-request-count 8 \
  --artifact-dir ~/runs/$RUN/probe-decode
```

The probes must not hit the prefix cache, or TTFT understates prefill. Check
that `vllm:prompt_tokens_cached` stays near 0 in their server metrics.

At concurrency 128, check that `max_num_seqs` isn't what limits the batch.
It isn't in the startup log.

```bash
# On the node, while the c=128 point runs. If running reaches 128, max_num_seqs
# isn't the limit. If it stays below 128 and waiting is above 0, check KV cache
# usage before blaming max_num_seqs.
curl -s localhost:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)'
```

After the probes, save the server log with the run:

```bash
# Save replica A's full log (startup facts and per-interval stats) with the run.
docker logs vllm-a > ~/runs/$RUN/server.log 2>&1
```

### What to read from a perf run

**From AIPerf's summary** (`profile_export_aiperf.csv` and `.json` for each
point):

- TTFT, ITL and request latency at p50, p90 and p99
- output tokens/s
- goodput
- failed requests

**From the server metrics** (`server_metrics_export.csv`). These names were
checked in the v0.33.0 image.

| Metric | What it tells you |
|---|---|
| `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` | Where each request's time went: waiting, in prefill, or in decode |
| `vllm:prompt_tokens`, `vllm:prompt_tokens_cached`, `vllm:request_prefill_kv_computed_tokens` | How much prefill the prefix cache saved, and how much was actually computed |
| `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` | The cache hit rate |
| `vllm:iteration_tokens_total` | Tokens per engine step: the real batch size, and how large the prefill chunks were |
| `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions` | Whether the batch is limited by slots or by KV space |
| `vllm:estimated_flops_per_gpu_total`, `vllm:estimated_read_bytes_per_gpu_total` | vLLM's own estimate of FLOPs and bytes per GPU. Divide by time and compare with B200 peaks to see how close each phase runs to its limit. Check whether they're filled in; they may need a flag |

### Diagnostics, when the gap analysis needs them

These are not part of every run.

**B2. Multi-turn with a shared prefix.** This isolates prefix caching. Each
run has 8-turn conversations, an 8k-token shared system prompt, and 24k tokens
of context per session.

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
  --concurrency 16 --artifact-dir ~/runs/$RUN/b2-multiturn
```

**B4. Speculative decoding acceptance.** Run this on DSpark runs (O4).

- **Dataset:** AIPerf 0.13 reads SPEED-Bench's coding split as a custom dataset
  (`--custom-dataset-type speed_bench_coding --input-file …`), so download that
  file first. `--public-dataset spec_al_humaneval` and `spec_al_mbpp` are
  coding sets that AIPerf downloads itself. `--help` lists dataset names with
  hyphens, but the parser accepts only the underscore spelling.
- **Acceptance:** read it from the server.

```bash
# Draft and accepted-token counters. Mean acceptance length is
# 1 + accepted_tokens / drafts.
curl -s localhost:8000/metrics | grep spec_decode
```

Never benchmark with synthetic acceptance. InferenceX's throughput mode fixes
the acceptance length (3.51) so runs are comparable, but those numbers don't
reflect real drafts. Use real rejection (`"rejection_sample_method":"block"`).

## 4. Quality run

A change that breaks the agent is not an optimization. Every run checks
quality right after its perf run, against the same replica and the same
configuration.

**A1. Smoke tests.** The answer to 17×19 must be 323
([getting-started step 11](getting-started.md#11-test-it-from-the-mac)). Then
send 20 tool-call requests (5 prompts × 4 repeats at the model card's
temperature 1.0). All 20 must return a well-formed `tool_calls` entry.

Run the check twice: once with `think=false`, and once with `think=true`.
A2 runs with thinking on, and thinking sends the output through the
reasoning parser before the tool parser.

```bash
# A1: 5 tool-call prompts x 4 repeats at temperature 1.0 against one replica.
# Prints a count of OK and FAIL; all 20 must be OK. Needs jq.
port=8000 think=false
for p in "List the files in the repository root." \
         "Show the last 20 lines of README.md." \
         "Find every Python file that imports requests." \
         "Run the unit tests." \
         "Show the five most recent git commits."; do
  for i in 1 2 3 4; do
    jq -n --arg p "$p" --argjson t $think '{
      model: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
      messages: [{role: "user", content: $p}],
      tools: [{type: "function", function: {name: "run_shell",
        description: "Run a shell command in the repository and return its output",
        parameters: {type: "object", properties: {command: {type: "string"}}, required: ["command"]}}}],
      tool_choice: "auto", temperature: 1.0, top_p: 1.0,
      chat_template_kwargs: {thinking: $t}, max_tokens: 8192}' \
    | curl -s localhost:$port/v1/chat/completions -H 'Content-Type: application/json' -d @- \
    | jq -r '.choices[0].message.tool_calls[0].function
             | if .name == "run_shell" and ((.arguments | fromjson | .command) | type) == "string"
               then "OK" else "FAIL" end' 2>/dev/null || echo FAIL
  done
done | sort | uniq -c
```

**A2. SWE-bench Verified, a fixed 50-task slice, with mini-SWE-agent 2.4.6.**
Use the same tasks, sampling settings and reasoning effort on every run.

```bash
# Install the agent and the SWE-bench harness (one time).
python3 -m venv ~/swe && . ~/swe/bin/activate && pip install mini-swe-agent==2.4.6 swebench==5.0.2

# Tell LiteLLM the local model's context size and zero cost, so cost tracking
# doesn't fail on an unknown model (one time).
cat > ~/swe/registry.json <<'EOF'
{"hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash-1017": {
  "max_input_tokens": 1048576, "max_output_tokens": 262144,
  "input_cost_per_token": 0, "output_cost_per_token": 0,
  "litellm_provider": "hosted_vllm", "mode": "chat"}}
EOF

# A model-only config layered over the default swebench.yaml (one time).
# Passing -c replaces the default config, so list swebench.yaml first.
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

# Run tasks 0-49 of SWE-bench Verified with 8 agents in parallel. Run from the
# run's folder, so the harness's logs and report land there too.
cd ~/runs/$RUN && . ~/swe/bin/activate
LITELLM_MODEL_REGISTRY_PATH=~/swe/registry.json mini-extra swebench \
  -c swebench.yaml -c ~/swe/dsv41.yaml \
  --subset verified --split test --slice 0:50 --workers 8 \
  --output ~/runs/$RUN/a2

# Score the patches locally. The harness runs each task's tests in Docker.
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path ~/runs/$RUN/a2/preds.json \
  --max_workers 16 --run_id $RUN
```

- **What one run can show:** with 50 tasks, the 95% confidence interval on a
  resolve rate is about ±13 points. A single run catches a broken change, such
  as tool calls failing or garbage output, but not a small regression. Most
  knobs here (scheduling, caching, CUDA graphs, layout, NUMA) shouldn't change
  the outputs beyond batch-dependent numerics. Expect a lossy knob, like fp8 →
  auto KV (O7), to be the one that moves the score.
- **Final vs baseline:** compare the two runs task by task (a paired
  comparison, McNemar's test), not as two headline numbers. This is the "small
  eval, big noise" problem from Kiely §5.1.3 (p. 128).
- **Sampling:** the model card says `temperature=1.0, top_p=1.0`, but the vLLM
  recipe says DeepSeek's published numbers used `top_p=0.95`. Pick one, record
  it, and keep it fixed.
- **Cost per task:** the SWE run is a real agentic load. Record its wall time
  and GPU-hours per resolved task. That gives you the async profile's cost per
  task.
- **Check the file format:** mini-SWE-agent writes `preds.json`. If the harness
  rejects it, convert it to JSONL first.

**A3. Speculative decoding is lossless.** Run this on DSpark runs (O4). Send
50 prompts at temperature 0 with DSpark off, then with DSpark on, and compare
the outputs. They should be identical or very nearly so. Investigate every
divergence. It is either batch-dependent numerics, which is expected and worth
understanding, or a bug.

## 5. Record

Raw output stays on the node, under `~/runs/<run>/`:

- `traces/`, `probe-prefill/`, `probe-decode/`: AIPerf's artifacts. Copy
  `traces/` into this repo's `runs/<run>/` too (3. Perf run, step 6). Git
  ignores `runs/`, so raw output isn't committed
- `server.log`: the replica's log
- `a2/`: the SWE-bench output

Summaries go in this repo:

- **`results/perf.csv`:** one row per run and point. Columns:
  - run, date, reference run, knob, phase, workload (traces, probe-prefill or
    probe-decode), concurrency
  - requests, failed requests
  - TTFT p50/p90/p99 and ITL p50/p90/p99 (ms)
  - request latency p90 (ms)
  - output tok/s per replica and per GPU
  - goodput
  - prefix cache hit rate
  - mean queue, prefill and decode time per request (ms)
- **`results/quality.csv`:** one row per run. Columns:
  - run, date
  - A1 passes out of 20, with thinking off and with thinking on
  - A2 tasks resolved out of 50, A2 wall time, GPU-hours per resolved task
- **`docs/experiments.md`:** one entry per run, in this format. The skills are
  written from these entries ([skills plan](skills-plan.md)).

```
## r<NN>: <knob> <reference value> → <new value> (phase: prefill | decode | pd)
- Reference run: r<MM>
- Why this knob: the gap from r<MM>'s analysis that it targets
- Prediction (and how it was made): …
- Startup facts: KV cache tokens, CUDA graph memory, start time
- Perf, vs the reference: TTFT and ITL p90 at c = 1, 8, 32; output tok/s per GPU; goodput
- Time split: queue / prefill / decode time per request; share of prompt tokens served from cache
- Probes: prefill tok/s at c = 1 and 4; ITL at c = 1 … 128
- Quality: A1 x/20 and x/20; A2 x/50 (reference y/50)
- Explanation (the metric or profile that shows why): …
- Decision: keep or revert. New reference: r<NN> or r<MM>
- Next gap → next knob: …
- Rule learned: "use X when …, avoid when …"
```

Before releasing the node, copy what you need from `~/runs/`.

## 6. Gap analysis

After each run, work through four steps. They pick the next knob.

1. **Find the biggest miss.** In the profile you're optimizing, which target
   fails, and at what concurrency? For the interactive profile that is TTFT or
   ITL at p90. For the async profile it is throughput per GPU.
2. **Split the time.** Use the server metrics to see where each request's time
   went: waiting in the queue, in prefill, or in decode.
3. **Match the symptom** in the table below. It names the phase and the knobs
   that act on it.
4. **Check the prediction.** If the result missed your prediction by more than
   about 25%, explain the miss before moving on. A profile (see
   [Analytical model and profiles](#analytical-model-and-profiles)) is the tool
   for that, and this step is where most of the learning happens.

| Symptom | Confirm with | Phase | Knobs |
|---|---|---|---|
| TTFT high, mostly queue time | Queue time ≫ prefill time; `num_requests_waiting` above 0 | Capacity: requests wait for a batch slot or KV space, which finishing decodes free | O9, O1 |
| TTFT high, mostly prefill time, few cache hits | `prompt_tokens_cached` small relative to `prompt_tokens` | Prefill: the cache isn't saving work | O6; KV capacity (O7, GPU memory utilization) |
| TTFT high, mostly prefill time, many cache hits | `request_prefill_kv_computed_tokens` still large; prefill probe below the analytical model | Prefill: compute | O5, O1 (more GPUs per prompt), O8 |
| ITL high even at concurrency 1–4 | Decode probe ITL at c=1 well above the analytical model | Decode: cost per step (bytes read, kernel launches, host reads) | O4, O1, O2, O3; CUDA graph coverage (O9) |
| ITL p50 fine, p99 spikes | Spikes line up with large steps in `iteration_tokens_total` (AIPerf's `--slice-duration` shows them over time) | Prefill interrupting decode | O5 (smaller chunks), O8 |
| ITL climbs steeply with concurrency | Decode probe ITL against batch size; `kv_cache_usage_perc` | Decode: batch growth (attention over long contexts, more experts read) | O1, O7, O9 |
| Low throughput per GPU (async) | `num_requests_waiting` above 0 while `num_requests_running` stops growing; preemptions. In the traces, running below `c` alone is expected, because sessions spend time in think-time gaps | Decode: the batch is capped | O9, O1, GPU memory utilization |

When two knobs fit the symptom equally, take the one listed first in its menu
below.

## Knob menus

### Prefill

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| O6 | Prefix caching | on → off (traces and B2 only) | How much of the agent's TTFT the cache is saving. Run it first, because it shows how much real prefill is left to optimize (Kiely §5.3.1, p. 136) |
| O5 | Chunked-prefill token budget | default (16,384 on v0.33.0) → `--max-num-batched-tokens` 2k / 4k (the recipe's value) / 8k / 32k | TTFT vs ITL: large chunks finish prefills sooner but stall the decodes sharing the batch (Kiely §5.5). This knob affects decode too, so record both metrics. It also sizes the activation memory vLLM reserves at startup (28.71 GiB per GPU at 16,384), which comes out of the KV cache |

### Decode

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| O1 | Parallel layout | 2×TP4 → 4×TP2 → 1×TP8 → DEP8 (`--data-parallel-size 8 --enable-expert-parallel`) | Per-GPU throughput vs per-user speed. TP2 is the recipe's B200 default. Run it with the baseline's other settings, then once as the full reference configuration in ADR 0002 (DSpark on, 8 sequences). TP8 halves per-GPU weights vs TP4 but doubles all-reduce traffic (Kiely §5.4.1, p. 144). The recipe lists DEP8 for B200 nodes; TP8 is **unverified**. The layout also changes prefill speed, so record TTFT too. A layout change restarts both replicas |
| O9 | Batch and CUDA graph sizes | default → `--max-num-seqs` 32 / 64 / 128; `--max-cudagraph-capture-size` ≥ batch × (1 + 5 drafts) | Where per-user speed falls off as the batch grows, and whether decode steps fall outside the captured graphs |
| O4 | Speculative decoding | off → DSpark with the reference configuration's `--speculative-config` (5 tokens, probabilistic drafts, block rejection, adaptive verification off), together with its `--compilation-config` and `--max-cudagraph-capture-size`. Then adaptive verification on. Then 1–5 tokens, with the capture sizes rebuilt as multiples of 1 + tokens | Acceptance length on coding traffic (B4), speedup vs concurrency, and A3. Expect the gain to shrink as batches grow (Kiely §5.2.1). Each decode step runs batch × (1 + drafts) tokens. With DSpark off, v0.33.0's default capture sizes reach 1,024 tokens, in steps of 8 and 16. That covers 170 requests at 6 tokens each, with padding: 7 × 6 = 42 tokens runs in the 48-token graph. vLLM may change the sizes when DSpark is on, and steps without a CUDA graph would understate DSpark's speedup, so check the capture sizes in the startup log |
| O2 | Engram placement | offload on → off. At TP4 the tables would add about 47 GiB per GPU (two 23.60 GiB tables per rank, from the startup log), which may fit | What host lookups cost per decode step, and how much KV cache they buy |
| O3 | NUMA pinning | none → each replica's container bound to its GPUs' socket: replica A `--cpuset-cpus 0-79 --cpuset-mems 0`, replica B `--cpuset-cpus 80-159 --cpuset-mems 1` | Step time and tail latency, given the per-step Engram reads from host memory |
| O7 | KV cache dtype | fp8 (baseline, from the recipe) → auto, if the attention backend accepts it | What fp8 KV costs in accuracy (A2) and buys in KV capacity |

### PD disaggregation

**O8. Compare two layouts on the same 8 GPUs.** Run it after the prefill and
decode menus, with their winners.

- **Aggregated (the reference):** two TP4 replicas, each doing both prefill and
  decode.
- **1P1D:** one TP4 prefill engine on GPUs 0–3 and one TP4 decode engine on
  GPUs 4–7. The prefill engine sends each request's KV cache to the decode
  engine over NVLink through vLLM's NIXL connector, behind
  `vllm-router --vllm-pd-disaggregation`.

Use the [recipe's PD commands](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
with these settings:

- the node's two GPU groups in place of its two GB200 trays
- a separate `VLLM_NIXL_SIDE_CHANNEL_PORT` for each engine
- each engine's flags carried over from the reference configuration
- if DSpark won O4, DSpark in **both** engines, so the KV cache stays
  compatible

The recipe verified 1P1D only on GB200 NVL4. On 8-GPU nodes it uses a whole
node per role, so **one-node 1P1D on B200 is unverified.** That is also open
question 6 in ADR 0002.

- **Predict first:**
  - **KV transfer cost.** This model's KV is small: about 890 bytes per token
    by DeepSeek's figure, or about 1.5 KB by vLLM's measured pool on this node
    (38 GiB per GPU for 26.75M tokens). A 100k-token prompt is then roughly
    0.1–0.2 GB, which should take well under a millisecond over NVLink. If so,
    transfer is not the cost to worry about.
  - **Decode capacity.** The real cost is that 1P1D has one decode engine
    where the aggregated layout has two. It wins only if removing prefill
    interference from decode is worth more than the lost decode capacity.
  - **Effect of the cache.** Coding-agent turns are mostly prefix-cache hits,
    so effective prefill per turn may be small. Expect gains mainly in ITL
    during bursts of cold, long prompts.
- **Measure:**
  - The perf run, against the router instead of replica A, at both target
    profiles.
  - TTFT and ITL at p50, p90 and p99. Look at the tail ITL while long
    prefills run.
  - KV transfer time (`vllm:nixl_xfer_time_seconds`).
  - Prefill-engine and decode-engine utilization. A large imbalance means 1:1
    is the wrong ratio for this traffic.
- **Decision rule:** adopt 1P1D for a profile only if it improves goodput or
  p99 ITL there without cutting async throughput per GPU by more than 10%.
  That threshold is a judgment call. Check the result against Kiely §5.5.2's
  criteria (p. 149).
- **Conditional disaggregation:** check whether `vllm-router` can send
  short or cache-hit prefills straight to the decode engine (Kiely §5.5.1,
  p. 148). If it can't, record that as a limitation.
- **If 1P1D won't start within half a day,** record the failure mode (error,
  versions, flags) and move on. A documented negative result still counts.

**The final configuration** combines whichever variants won, measured
together as one run. Combinations interact, so measure them; don't add up the
individual gains.

## Analytical model and profiles

- **Predict before run 0:**
  - Decode step time at each of the decode probe's batch sizes, from bytes
    read per step ÷ HBM bandwidth.
  - Prefill time for the prefill probe, from FLOPs ÷ achievable compute. This
    model's prefill runs only the 20-layer encoder half, about 8B active
    parameters per token, against about 16B per token in decode.
  - The compute ceiling for the experts. vLLM v0.33.0 picks
    `FLASHINFER_TRTLLM_MXFP4_MXFP8` on this node: MXFP4 weights with MXFP8
    activations, so the experts run at the FP8 rate. Dense per GPU: ~9 PFLOPS
    at FP4, ~4.5 at FP8, ~2.25 at BF16 (ADR 0002, Hardware).

  Then compare the predictions with the probes. The target is decode within
  about 25% at five or more batch sizes. That is a judgment call, as in ADR
  0001. Kiely §2.4 (pp. 61–64) shows how to prove which resource bounds each
  phase: compare an operation's arithmetic intensity with the GPU's ops:byte
  ratio.
- **Profile two configurations,** one decode-heavy (the decode probe at
  concurrency 64 or 128) and one prefill-heavy (the prefill probe at
  concurrency 1). Use vLLM's torch profiler: check `vllm serve --help` in v0.33
  for the profiler flag, then call `/start_profile` and `/stop_profile` around
  a short run. AIPerf's `--server-profiler` flag can make those calls for you.
  Add Nsight Systems if it's available. Attribute the time to:
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
- [ ] **Run 0 complete.** The analytical model's predictions are written down
  first. The trace sweep has 3 repeats and zero failed requests. Both probes
  have run, A1 passes, and the A2 resolve rate is recorded.
- [ ] **Every run went through the loop.** Each has a prediction, perf and
  quality results, a gap analysis and a decision in `docs/experiments.md`, and
  rows in `results/`.
- [ ] **Prefill.** O6 and O5 are done, with their effect on both TTFT and ITL.
- [ ] **Decode.** O1, O9 and O4 are done, plus at least one of O2, O3 and O7.
- [ ] **PD disaggregation.** O8's 1P1D layout is measured against the
  aggregated 2×TP4 layout at both target profiles, and the decision rule is
  applied. If 1P1D couldn't run, the failure is documented.
- [ ] **Final configuration.** It beats the baseline on interactive goodput
  (concurrency per replica meeting both targets) **or** on async throughput per
  GPU. Report both deltas. The target is ≥ 1.3× on at least one, which is a
  judgment call.
- [ ] **Quality on the final configuration.** A1 passes; A2 shows no
  regression in the paired comparison with run 0; A3 shows DSpark is
  lossless, or the divergences are explained.
- [ ] **Target table.** For both baseline and final: the maximum concurrency
  meeting the interactive targets, throughput per GPU under the async targets,
  cost per 1M output tokens, and cost per resolved SWE task.
- [ ] **Analytical model.** Predicted vs measured decode step time at five or
  more batch sizes, with every gap over 25% explained by a profile.
- [ ] **Two profiles** (decode-heavy and prefill-heavy) with time attribution.
- [ ] **Packaged endpoint.** The final configuration as a single launch script
  (or compose file) plus a README listing its targets, measured results, and
  known limits.
- [ ] **Write-up.** About two pages, with four charts:
  - throughput per GPU vs tok/s per user (the Pareto curve)
  - TTFT vs concurrency
  - DSpark acceptance and speedup vs concurrency
  - aggregated vs 1P1D: TTFT and p99 ITL on the traces
- [ ] **Node released** on or before Fri 2026-11-20.

**Optional, if time allows:** a comparison with the same model on B300 at the
same targets.

**After the PoC:** turn the experiment log into two shareable Claude skills,
one for prefill and one for decode. See the [skills plan](skills-plan.md).

## Later: image input on one TP8 replica

The PoC is text only. Every configuration above runs with
`--language-model-only`, so vLLM never loads the vision encoder. Image input
is a later step. It isn't part of the done criteria, and it runs only after
the text work, if node time is left before the release date.

- **Why one TP8 replica:** a single replica on all 8 GPUs has the most HBM for
  the extra KV cache that image tokens need, and for the memory vLLM reserves
  for worst-case images at startup. O1's 1×TP8 point is the text-only
  baseline to compare against. If TP8 doesn't start in O1, use one TP4
  replica instead.
- **Change:** take O1's TP8 configuration and replace `--language-model-only`
  with `--limit-mm-per-prompt '{"image": N}'`, where N is the most images a
  request needs. Start with 1.
- **Measure:**
  - The cost of turning images on: KV cache memory and token capacity from the
    startup log, and startup time, with and without the flag.
  - Text traffic with images on: the traces and the decode probe at a few
    concurrencies, compared with O1's TP8 point. Text-only requests shouldn't
    get slower.
  - Image requests: TTFT and prompt tokens per image at a few resolutions.
    config.json caps an image at 1,024 tokens. For load, use a mixed workload,
    such as agent turns that include a screenshot. Check
    `aiperf profile --help` for its image options.
  - Repeated images: send the same screenshot again in later turns and check
    that prefix caching reuses it.
- **Quality:** a smoke test with an image whose answer is known, such as a
  screenshot of an error message that the reply has to quote. This also
  confirms that `--tokenizer-mode deepseek_v41` passes images through, which
  hasn't been verified for this model.

## Schedule

| Dates | Work |
|---|---|
| Fri 11-06 – Sun 11-08 | **Run 0.** Write the analytical model's predictions. Analyze the traces. Then the perf run with 3 repeats, the probes, A1 and A2 |
| Mon 11-09 – Sun 11-15 | **Prefill and decode runs,** one knob per run, picked by each gap analysis. Profile the prefill-heavy and decode-heavy configurations. At 4–6 h per run, expect 2 runs a day, if the node keeps running overnight |
| Mon 11-16 – Tue 11-17 | **PD disaggregation:** O8, 1P1D vs 2×TP4 aggregated |
| Wed 11-18 – Fri 11-20 | **Final configuration run,** A3 on it. Package, write up, release the node |

If runs take longer than planned, cut O7, then O3, then O2, before cutting PD
disaggregation or a quality check.

## Keep out of the repo

Customer names, deal pricing, and anything quoted from internal channels.
