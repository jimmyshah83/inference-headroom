# PoC: learn prefill and decode on 1×B300

**The goal is to understand prefill and decode in depth by measuring them.**
The vehicle is a real endpoint: DeepSeek-V4-Flash-0731 on vLLM, serving a
coding-agent workload on one B300 GPU.

The plan has four steps:

1. **Set targets** from the slowest provider of this model on Artificial
   Analysis, treated as the customer's bar ([Targets](#targets-the-slowest-provider-on-artificial-analysis)).
2. **Run the baseline sweep** on the coding-agent traces, concurrency 1–32,
   once ([3. Perf run](#3-perf-run)).
3. **Read the results** from AIPerf's metrics table, and choose the operating
   concurrency: the highest one that still meets the targets
   ([Choose the operating concurrency](#choose-the-operating-concurrency)).
4. **Optimize prefill and decode one knob at a time** at that concurrency, to
   see how far toward the fastest provider the server can get
   ([The loop](#the-loop)).

The window runs to **Fri 2026-11-20**.

- Setup: [getting-started.md](getting-started.md)
- Decisions: [ADR 0003](adr/0003-deepseek-v4-flash-0731-on-1xb300-with-vllm.md)
- Log: `docs/experiments.md`, one entry per run

## Prefill and decode

Every request runs in two phases (Kiely §2.4.2, p. 63):

| | Prefill | Decode |
|---|---|---|
| What it does | Runs the whole prompt through the model in one pass and writes its KV cache | Generates one token per step. Each step reads the active weights and the KV cache |
| Bound by | Compute (FLOPs) | Memory bandwidth (bytes per step), until the batch is large |
| The user sees it as | Time to first token (TTFT) | Inter-token latency (ITL), or tokens/s per user |
| Ways to make it cheaper on one GPU | Skip it (prefix caching), size the chunks | Batch more requests per step, read fewer bytes per step, make more tokens per step (speculative decoding), launch less (CUDA graphs) |

The two phases share the GPU. When a prefill chunk joins a batch, every
decode in that step waits for it. That coupling is why some changes, like the
chunk size, trade TTFT against ITL.

A coding agent adds one more metric. A tool call is useful only once it is
complete, so the agent waits for the whole response, not just the first token
(Kiely §1.4, p. 36). Record request latency as well as TTFT and ITL.

## Targets: the slowest provider on Artificial Analysis

Artificial Analysis benchmarks the public API providers of DeepSeek V4 Flash
0731 ([providers page](https://artificialanalysis.ai/models/deepseek-v4-flash/providers),
[methodology](https://artificialanalysis.ai/methodology/performance-benchmarking)).
The PoC treats the **slowest provider as the customer's floor** and the
**fastest as the stretch goal**.

Use its **multi-turn agentic workload**, not the headline numbers. It is
agentic tasks with tool calls, which is closest to the coding-agent traces.
The headline averages it with single 1k-, 10k- and 100k-token prompts. The
worst headline TTFT (CoreWeave, 4.65 s) comes mostly from its 100k-token
prompts: its multi-turn TTFT is 0.61 s.

Values read on 2026-11-08, from 10 providers with measurements:

| Metric | Floor (slowest) | Stretch (fastest) | AIPerf row to compare |
|---|---|---|---|
| TTFT, median | ≤ **2.52 s** (GMI) | 0.26 s (Baseten) | TTFT p50 |
| Output speed per user, median | ≥ **31 tokens/s** (SiliconFlow) | 320 tokens/s (Baseten) | E2E Output TPS/User p50 |

How the numbers line up with AIPerf's:

- **Both are medians.** Each Artificial Analysis value is the average of
  per-run medians over 5 days, so compare it with AIPerf's p50.
- **Output speed includes TTFT.** Artificial Analysis divides output tokens,
  reasoning included, by the time from sending the request to the last token.
  That is AIPerf's E2E Output TPS/User, not Output TPS/User.
- **TTFT is the first token, reasoning included,** as in AIPerf.
- **Their numbers include the network** from Google Cloud us-central1. AIPerf
  runs on the node, so it doesn't. Treat a close call as a miss.
- **Different prompts and load.** Their agentic tasks aren't these traces, and
  their load level isn't published. So the targets are reference points that
  make the goal concrete, not a head-to-head comparison.

Check the values again before the write-up; they move. Price and uptime are
out of scope.

## The loop

The baseline (`r00-baseline`) runs the full sweep once. From it, choose the
**operating concurrency C**. Every later run changes **one knob** from the
current reference configuration and measures one point, at C. If the change
helps and passes the quality checks, it becomes the new reference.

```
r00-baseline sweep ─> choose C ─> 1 Predict ─> 2 Start ─> 3 Perf at C ─> 4 Quality ─> 5 Record ─> 6 Gap analysis
                                       ^                                                              │
                                       └──────────── next knob, from the prefill or decode menu ──────┘
```

1. **Predict.** Before you measure, write down which metric should move, in
   which direction, by about how much, and why. Use the
   [analytical model](#analytical-model-and-profiles) where you can.
2. **Start.** Restart the server with the change. Record what its startup log
   reports: KV cache tokens, CUDA graph memory, and the kernels it picked
   ([getting-started step 8](getting-started.md#8-start-the-server)).
3. **Perf.** AIPerf replays the coding-agent traces at C, then runs two short
   phase probes. See [3. Perf run](#3-perf-run).
4. **Quality.** Run the A1 smoke tests, then a 50-task SWE-bench slice. See
   [4. Quality run](#4-quality-run).
5. **Record.** Add rows to `results/perf.csv` and `results/quality.csv`, and an
   entry to `docs/experiments.md`. See [5. Record](#5-record).
6. **Gap analysis.** Compare the run with the targets, the prediction and the
   reference run. Decide which phase holds the biggest gap, and pick the next
   knob from that phase's menu. See [6. Gap analysis](#6-gap-analysis).

**How far you got** is measured at the end: the final configuration runs the
full sweep again, which shows how much higher the concurrency that meets the
floor went, and how close to the stretch goal TTFT and output speed got.

**Naming.** Name each run `r<NN>-<knob>-<value>`, for example
`r01-p1-prefix-cache-off`. Raw output stays on the node under `~/runs/<run>/`,
and the results are analyzed there.

**Rules** (carried over from ADR 0001):

- Pin the image, model revision and flags. Append to `~/runs/env.txt` every
  session ([getting-started step 10](getting-started.md#10-record-the-environment-then-stop)).
- Warm up before measuring. The trace scenario does this itself.
- Change one variable at a time, in the server and in the benchmark (Kiely
  §4.5.2, p. 114). Re-run the baseline whenever the image or driver changes.
- Report failed requests. Any error invalidates the point.

**Time per run, estimated:**

| Step | Time |
|---|---|
| Start | Measure it on the baseline |
| Perf: one point at C | ~20 min |
| Perf: the baseline sweep (6 points) | ~2¼ h on the 8×B200 node; measure it here |
| Perf: phase probes | ~20 min |
| Quality: A1 | ~5 min |
| Quality: A2 | Measure it on the baseline. Estimate: 1.5–3 h |

Keep the node running between runs: a stopped node costs a restart, and A2
fits best overnight.

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

Before the baseline, run `aiperf analyze-trace` on the trace dataset and
record its input-length distribution and prefix-reuse rate. It needs no
server, so run it on the Mac. Those numbers tell you which phase can matter.

## 3. Perf run

Run AIPerf on the node, inside tmux, for every run. From the Mac, every request
would go through the SSH tunnel: TTFT would include the round trip, and the
tunnel can bunch tokens and skew ITL. On 2026-11-06 a sweep from the Mac also
got `Server disconnected` errors that the server never saw. tmux keeps a sweep
running when the SSH connection drops or the laptop sleeps.

```bash
# On the Mac (one time): install the pinned AIPerf from pyproject.toml into the
# repo's .venv, for `aiperf analyze-trace` and --dry-run.
uv sync && . .venv/bin/activate

# On the node (one time): install AIPerf into its own venv. If venv is missing,
# install it first (on Ubuntu 24.04: sudo apt-get install -y python3.12-venv).
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

These are SemiAnalysis's Claude Code traces, the dataset InferenceX uses for
its agentic sweeps. They are replayed under InferenceX's AgentX scenario
(`--scenario inferencex-agentx-mvp`), AIPerf's locked method for this dataset.
The concurrencies are InferenceX's: 1, 2, 4, 8, 16 and 32.

**Every run uses exactly this workload.**
[`scripts/sweep_traces.py`](../scripts/sweep_traces.py) holds the AIPerf flags
as fixed constants. Only the run name, the concurrency and the server change
between runs, so a difference in the results comes from the server. If a
constant has to change, run the baseline again with the new value.

The script runs one AIPerf profile per concurrency, because the scenario
refuses a sweep in one command. Each point primes its trajectories' caches,
warms up for 2 more minutes, then measures for 15 minutes. The grace period
lets responses that started before the deadline finish and count. On the
8×B200 node each point took 17–27 minutes.

#### Run the baseline sweep: step by step

Each step says where it runs. The sweep runs inside tmux on the node, so it
keeps going when the SSH connection drops or the laptop closes. The tmux
commands it needs:

| To | Run |
|---|---|
| Start the session (on the node, not the Mac) | `tmux new -s sweep` |
| Leave it running (detach) | Ctrl-b, then d. If your terminal catches Ctrl-b, close the terminal tab instead, or run `ssh b300-usn1-1 tmux detach-client -s sweep` from another Mac terminal |
| Come back to it | `ssh b300-usn1-1`, then `tmux attach -t sweep` |
| List the sessions | `tmux ls`. `(attached)` means a terminal is showing it |
| Stop the sweep | Ctrl-c inside tmux. Rerunning the same command resumes it |
| Close the session | `exit` at the tmux prompt, once the sweep has finished |

**1. Mac, repo root: copy the script to the node.** The repo is private, so
the node can't clone it. Copy it again whenever the script changes.

```bash
scp scripts/sweep_traces.py b300-usn1-1:~/
```

**2. Mac: log in to the node and start tmux.** The prompt must show the
node's hostname before you start tmux. The Mac doesn't have tmux.

```bash
ssh b300-usn1-1
tmux new -s sweep
```

**3. tmux: set up the shell.** `~/.bashrc` sets `HF_HOME=~/hf` in every new
shell, so set it again here. The last command must print
`/home/jimmyshah-sshkey/hf-aiperf`.

```bash
. ~/aiperf/bin/activate
export HF_HOME=~/hf-aiperf
echo $HF_HOME
```

**4. tmux: check the commands, then test one point (about 20 minutes).** The
dry run prints the AIPerf command without running it. It must contain
`--model deepseek-ai/DeepSeek-V4-Flash-0731`. If it shows another model, the
node has an old copy of the script: repeat step 1. Always pass
`--runs-dir ~/runs`. Without it, the script tries to write to `/home/runs`,
which it can't create.

```bash
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs --concurrency 1 --dry-run
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs --concurrency 1
```

**5. tmux: check the test point.** This lists points that had errors or that
AIPerf cancelled. It must print nothing. If it lists the point, read its
`logs/aiperf.log` and fix the cause before going on.

```bash
grep -l -E "Error Summary|INCOMPLETE" ~/runs/r00-baseline/traces/rep*/c*/profile_export_console.txt
```

**6. tmux: run the full sweep, once.** Without `--concurrency`, the script
runs all six concurrencies, 1, 2, 4, 8, 16 and 32, one after another. The c1
point from step 4 is already done, so it isn't measured again.

```bash
# The baseline sweep: c = 1, 2, 4, 8, 16, 32, one pass.
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs
```

**7. tmux: detach, so the sweep keeps running without you.** Press Ctrl-b,
let go, then press d. tmux prints `[detached (from session sweep)]` and you're
back at the node's normal prompt. Now `exit` closes only the SSH connection,
and you can close the laptop.

If Ctrl-b does nothing, your terminal is catching it. Close the terminal tab
(tmux keeps running), or detach from another Mac terminal:

```bash
ssh b300-usn1-1 tmux detach-client -s sweep
```

**8. Mac: attach, to check on the sweep.**

```bash
ssh b300-usn1-1
tmux ls                                         # should list "sweep"
cat ~/runs/r00-baseline/traces/progress.tsv     # one line per finished point
tmux attach -t sweep                            # the live output
```

If `tmux ls` says `no server running`, the session is gone: check
`progress.tsv`, then start again from step 2. The sweep resumes where it
stopped. When you're done looking, detach again as in step 7.

**9. tmux: check every point when the sweep ends.** Run the step 5 command
again. It must print nothing. AIPerf exits 0 and writes a summary even when it
cancels a point, so the script counts that point as done and skips it on a
rerun. For each point listed, read its `logs/aiperf.log`, delete its folder
(for example `rm -r ~/runs/r00-baseline/traces/rep1/c8`), and rerun the
step 6 command. It measures only the missing points.

#### Choose the operating concurrency

Read every point's table ([What to read](#what-to-read-from-a-perf-run)) and
put the six side by side. As concurrency rises, expect:

- **Total output (Output TPS)** to climb, because each step serves more
  requests. That's batching.
- **Speed per user (ITL, Output TPS/User, E2E Output TPS/User)** to fall,
  because each decode step carries more requests.
- **TTFT** to rise slowly, then jump once requests wait for each other's
  prefill.

The point where total output stops climbing but latency keeps rising is the
server's practical limit. Choose **C as the highest concurrency where TTFT p50
≤ 2.52 s and E2E Output TPS/User p50 ≥ 31**, the floor from the
[targets](#targets-the-slowest-provider-on-artificial-analysis). Write C and
the reason in `docs/experiments.md`.

- **If c32 still meets the floor,** there's headroom above the sweep. Add a
  point: `--concurrency 64`.
- **If c1 already misses it,** start from c1 and optimize toward the floor.

Every later run measures one point at C:

```bash
python ~/sweep_traces.py r01-p1-prefix-cache-off --runs-dir ~/runs --concurrency <C>
```

**Noise (recommended).** One run wobbles: on the 8×B200 node, output
throughput varied by 5–30% between identical repeats. Repeat the baseline at
C twice more, which adds `rep2` and `rep3` (about 40 minutes):

```bash
python ~/sweep_traces.py r00-baseline --runs-dir ~/runs --concurrency <C> --repeats 3
```

If a later run differs from the baseline by less than twice the spread between
those repeats, repeat it before you believe it (Kiely §4.5.2, p. 114).

#### What the script and the scenario do

The script:

- **Checks the server first.** It calls `/health` on the URL (default
  `http://localhost:8000`) and stops if the server is down.
- **Resumes.** A point that already has its summary
  (`profile_export_aiperf.json`) is skipped, so rerunning the same command
  picks up an interrupted sweep. A cancelled point has a summary too (step 9).
- **Stops cleanly.** Ctrl-C stops the whole sweep, not just the current point.
- **Logs progress.** It appends one line per point (start, end, exit code) to
  `<runs-dir>/<run>/traces/progress.tsv`.
- **Picks the model with `--model`.** The default is DeepSeek-V4-Flash-0731.
  It must match the model the server serves.

AIPerf can repeat a point itself (`--num-profile-runs`), but the sweep doesn't
use it. Its repeats run back-to-back, and by default only the first repeat
warms up, so in agentic replay the later repeats start from a different cache
state.

What the scenario does (from AIPerf 0.13.0's `inferencex_agentx_mvp.py`):

- **Agentic replay.** Each of the `c` sessions starts its trace at a random
  point (0–100% of the way through), and its context is primed before
  measuring. Each point therefore measures a steady state, not every session
  starting cold at turn 0. Sub-agent fan-out (the Task tool) is replayed too.
- **Recorded think times.** The gaps between an agent's turns (tool runs) are
  kept, each capped at 10 s. `c` is the number of concurrent agent sessions,
  not of requests in flight. On the 8×B200 node, c8 averaged about 2 requests
  in flight and c32 about 8. AIPerf's Effective Concurrency row shows the real
  number.
- **Exact output lengths.** It adds `ignore_eos`, so each turn generates
  exactly its recorded output length. Otherwise a change that alters what the
  model writes, like the KV cache dtype, would also change how long decode
  takes.
- **Locks the method.** It requires streaming, at least 900 s per point, and
  the `first_turn_prefix` cache-bust marker (which it doesn't set for you).
  It rejects conflicting flags unless `--unsafe-override` is passed, which
  marks the run invalid. Don't use it for recorded runs.

The other flags hold the rest of the workload fixed:

| Flag | Without it |
|---|---|
| `--random-seed 42` | The scenario picks a fresh random seed for each run, so the traces and their starting points would differ from run to run |
| `--cache-bust first_turn_prefix`, and one AIPerf run per concurrency | The prefix cache keeps its contents between points. Without a marker, each point would replay traces the previous point already cached, so later points would look faster. The marker is built from an ID that AIPerf creates fresh for every run, so separate runs don't share cached conversations. It goes before the first user turn, so the shared system prompt can still hit the cache, as it would in real use |
| `--tokenizer-revision` | AIPerf builds the prompts with the tokenizer. A change to the tokenizer on Hugging Face could change prompt lengths |

What each point writes to `~/runs/<run>/traces/rep<r>/c<c>/`:

- `profile_export_console.txt`: AIPerf's metrics table, as shown live
- `profile_export_aiperf.csv` and `.json`: the same summary, as data
- `profile_export.jsonl`: one line per request
- `server_metrics_export.csv` and `.json`: the server's `/metrics`, scraped
  during the run. This is where the split between prefill and decode comes
  from
- `logs/aiperf.log`: check it first when a point fails

If AIPerf refuses a concurrency because the trace pool is too small, the error
says so. `--allow-dataset-wrap` reuses traces, but reused traces also hit the
prefix cache, so record it if you use it.

### What to read from a perf run

**After each point, read AIPerf's table.** When a point finishes, AIPerf saves
the same metrics table it shows live in `profile_export_console.txt`, in that
point's folder. The same numbers are in `profile_export_aiperf.csv` and
`.json`.

```bash
# One point's table. less -S keeps the wide rows on one line; scroll with the
# arrow keys and quit with q.
less -S ~/runs/$RUN/traces/rep1/c1/profile_export_console.txt

# Every point of the run, one after another.
for f in ~/runs/$RUN/traces/rep1/c*/profile_export_console.txt; do
  echo "== $f"; cat "$f"
done | less -S
```

The live dashboard's top rows (KV cache usage, server running and waiting
requests, preemptions, server throughput) are snapshots of the server's
`/metrics` and aren't in this file. After the run, read them from
`server_metrics_export.csv` in the same folder (see the second table below).

Start with the two target rows: **TTFT p50** against 2.52 s, and **E2E Output
TPS/User p50** against 31 tokens/s. Then read the rest to see why. Compare
avg with p50: a right-skewed latency distribution pulls the average above the
median, so read p50 and p90 together (Kiely §1.4.1, p. 37).

| Row | What it tells you | Phase |
|---|---|---|
| ISL, OSL (tokens) | Prompt and answer sizes: the shape of the workload | |
| Theoretical Prefix Cache Hit (%) | The share of prompt tokens that repeat an earlier prompt, so prefill can skip them. ISL × (1 − hit rate) is roughly what prefill still computes per request. The server's actual hit rate is in its metrics | Prefill |
| TTFT (ms) | The wait for the first token: queueing plus prefill of the uncached part | Prefill |
| ITL (ms), Output TPS/User | Decode speed for one user. Output TPS/User ≈ 1000 ÷ ITL | Decode |
| E2E Output TPS/User | Output tokens ÷ the whole request time, TTFT included. This is what OpenRouter and Artificial Analysis call output speed | Both |
| Req Latency, Decode Duration (ms) | The whole request, and how much of it was decode. A coding agent waits for the whole response | Both |
| Output TPS, Req/sec | The server's total output, across all users | Throughput |
| Goodput | Requests per second that met the script's own thresholds (TTFT ≤ 3 s, ITL ≤ 20 ms). They aren't the targets above | |
| Err %, Completed | Any failed request invalidates the point | |
| Effective Concurrency (requests) | Requests actually in flight. In the traces, sessions pause between turns, so this is below the concurrency you set | Load |
| Effective Prefill / Decode Concurrency | How many requests were in prefill, and how many in decode, on average | Both |

**From the server metrics** (`server_metrics_export.csv`). These names were
checked in the v0.33.0 image.

| Metric | What it tells you |
|---|---|
| `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` | Where each request's time went: waiting, in prefill, or in decode |
| `vllm:prompt_tokens`, `vllm:prompt_tokens_cached`, `vllm:request_prefill_kv_computed_tokens` | How much prefill the prefix cache saved, and how much was actually computed |
| `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` | The server's actual cache hit rate |
| `vllm:iteration_tokens_total` | Tokens per engine step: the real batch size, and how large the prefill chunks were |
| `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions` | Whether the batch is limited by slots or by KV space |
| `vllm:estimated_flops_per_gpu_total`, `vllm:estimated_read_bytes_per_gpu_total` | vLLM's own estimate of FLOPs and bytes. Divide by time and compare with the B300's peaks to see how close each phase runs to its limit. Check whether they're filled in; they may need a flag |

### Phase probes

The traces mix prefill and decode. Two short synthetic runs measure each phase
on its own, so you can see which phase a change moved and compare it with the
analytical model. `$MODEL` comes from `~/.bashrc` ([getting-started step 7](getting-started.md#7-pull-the-pinned-vllm-image-and-download-the-model)).

```bash
# Prefill probe: 64k-token prompts, one output token, one and four at a time.
# TTFT here is almost all prefill, so prefill throughput = 64,000 / TTFT.
aiperf profile --model $MODEL \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer $MODEL \
  --isl 64000 --osl 1 \
  --extra-inputs '{"ignore_eos": true, "chat_template_kwargs": {"thinking": false}}' \
  --concurrency 1,4 --benchmark-duration 120 --benchmark-grace-period 120 \
  --warmup-request-count 2 \
  --artifact-dir ~/runs/$RUN/probe-prefill

# Decode probe: 1k in, exactly 1k out, at five batch sizes. ITL here is the
# time of one decode step at that batch size, which the analytical model
# predicts.
aiperf profile --model $MODEL \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer $MODEL \
  --isl 1000 --osl 1000 \
  --extra-inputs '{"ignore_eos": true, "chat_template_kwargs": {"thinking": false}}' \
  --concurrency 1,4,16,64,128 --benchmark-duration 180 --benchmark-grace-period 120 \
  --warmup-request-count 8 \
  --artifact-dir ~/runs/$RUN/probe-decode
```

The probes must not hit the prefix cache, or TTFT understates prefill. Check
that `vllm:prompt_tokens_cached` stays near 0 in their server metrics.

At concurrency 128, check that `max_num_seqs` isn't what limits the batch:

```bash
# On the node, while the c=128 point runs. If running reaches 128, max_num_seqs
# isn't the limit. If it stays below 128 and waiting is above 0, check KV cache
# usage before blaming max_num_seqs.
curl -s localhost:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)'
```

After the probes, save the server log with the run:

```bash
# Save the server's full log (startup facts and per-interval stats) with the run.
docker logs vllm-a > ~/runs/$RUN/server.log 2>&1
```

### Diagnostics, when the gap analysis needs them

These are not part of every run.

**B2. Multi-turn with a shared prefix.** This isolates prefix caching. Each
run has 8-turn conversations, an 8k-token shared system prompt, and 24k tokens
of context per session.

```bash
# 64 conversations of ~8 turns each, back to back, at concurrency 16.
aiperf profile --model $MODEL \
  --url http://localhost:8000 --endpoint-type chat --streaming \
  --tokenizer $MODEL \
  --conversation-num 64 --conversation-turn-mean 8 \
  --shared-system-prompt-length 8000 \
  --user-context-prompt-length 24000 --num-dataset-entries 64 \
  --isl 1500 --osl 500 \
  --extra-inputs '{"chat_template_kwargs": {"thinking": false}}' \
  --concurrency 16 --artifact-dir ~/runs/$RUN/b2-multiturn
```

**B4. Speculative decoding acceptance.** Run this on D1 runs.

- **Dataset:** `--public-dataset spec_al_humaneval` and `spec_al_mbpp` are
  coding sets that AIPerf downloads itself. AIPerf 0.13 also reads
  SPEED-Bench's coding split as a custom dataset
  (`--custom-dataset-type speed_bench_coding --input-file …`); download that
  file first.
- **Acceptance:** read it from the server.

```bash
# Draft and accepted-token counters. Mean acceptance length is
# 1 + accepted_tokens / drafts.
curl -s localhost:8000/metrics | grep spec_decode
```

Never benchmark with synthetic (fixed) acceptance. It makes runs comparable,
but the numbers don't reflect real drafts.

## 4. Quality run

A change that breaks the agent is not an optimization. Every run checks
quality right after its perf run, against the same server and configuration.

**A1. Smoke tests.** The answer to 17×19 must be 323
([getting-started step 9](getting-started.md#9-test-it-from-the-mac)). Then
send 20 tool-call requests (5 prompts × 4 repeats at DeepSeek's agentic
sampling: temperature 1.0, top_p 0.95). All 20 must return a well-formed
`tool_calls` entry.

Run the check twice: once with `think=false`, and once with `think=true`
(effort `high`). A2 runs with thinking on, and thinking sends the output
through the reasoning parser before the tool parser.

```bash
# A1: 5 tool-call prompts x 4 repeats against the server. Prints a count of OK
# and FAIL; all 20 must be OK. Needs jq.
port=8000 think=false
for p in "List the files in the repository root." \
         "Show the last 20 lines of README.md." \
         "Find every Python file that imports requests." \
         "Run the unit tests." \
         "Show the five most recent git commits."; do
  for i in 1 2 3 4; do
    jq -n --arg p "$p" --argjson t $think '{
      model: "deepseek-ai/DeepSeek-V4-Flash-0731",
      messages: [{role: "user", content: $p}],
      tools: [{type: "function", function: {name: "run_shell",
        description: "Run a shell command in the repository and return its output",
        parameters: {type: "object", properties: {command: {type: "string"}}, required: ["command"]}}}],
      tool_choice: "auto", temperature: 1.0, top_p: 0.95,
      chat_template_kwargs: (if $t then {thinking: true, reasoning_effort: "high"} else {thinking: false} end),
      max_tokens: 8192}' \
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
# doesn't fail on an unknown model (one time). 393,216 output tokens is the
# recipe's allowance for high and max effort.
cat > ~/swe/registry.json <<'EOF'
{"hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731": {
  "max_input_tokens": 1048576, "max_output_tokens": 393216,
  "input_cost_per_token": 0, "output_cost_per_token": 0,
  "litellm_provider": "hosted_vllm", "mode": "chat"}}
EOF

# A model-only config layered over the default swebench.yaml (one time).
# Passing -c replaces the default config, so list swebench.yaml first.
cat > ~/swe/dsv4.yaml <<'EOF'
model:
  model_name: "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash-0731"
  model_kwargs:
    api_base: "http://localhost:8000/v1"
    temperature: 1.0
    top_p: 0.95
    extra_body:
      chat_template_kwargs: {thinking: true, reasoning_effort: "high"}
EOF

# Run tasks 0-49 of SWE-bench Verified with 8 agents in parallel. Run from the
# run's folder, so the harness's logs and report land there too.
cd ~/runs/$RUN && . ~/swe/bin/activate
LITELLM_MODEL_REGISTRY_PATH=~/swe/registry.json mini-extra swebench \
  -c swebench.yaml -c ~/swe/dsv4.yaml \
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
  knobs here (scheduling, caching, CUDA graphs, the MoE kernel) shouldn't
  change the outputs beyond batch-dependent numerics. Expect a precision knob,
  like the KV cache dtype (D4), to be the one that moves the score.
- **Final vs baseline:** compare the two runs task by task (a paired
  comparison, McNemar's test), not as two headline numbers. This is the "small
  eval, big noise" problem from Kiely §5.1.3 (p. 128).
- **Sampling:** DeepSeek recommends temperature 1.0 with top_p 0.95 for agentic
  use. Keep it fixed.
- **Cost per task:** the SWE run is a real agentic load. Record its wall time
  and GPU-hours per resolved task.
- **Check the file format:** mini-SWE-agent writes `preds.json`. If the harness
  rejects it, convert it to JSONL first.

**A3. Speculative decoding is lossless.** Run this on D1 runs. Send 50
prompts at temperature 0 with speculative decoding off, then on, and compare
the outputs. They should be identical or very nearly so. Investigate every
divergence. It is either batch-dependent numerics, which is expected and worth
understanding, or a bug.

## 5. Record

Raw output stays on the node, under `~/runs/<run>/`:

- `traces/`, `probe-prefill/`, `probe-decode/`: AIPerf's artifacts
- `server.log`: the server's log
- `a2/`: the SWE-bench output

Summaries go in this repo:

- **`results/perf.csv`:** one row per run and point. Columns:
  - run, date, reference run, knob, phase, workload (traces, probe-prefill or
    probe-decode), concurrency
  - requests, failed requests
  - TTFT p50/p90/p99 and ITL p50/p90/p99 (ms)
  - E2E output tokens/s per user, p50
  - request latency p90 (ms)
  - output tokens/s (total)
  - goodput
  - effective concurrency
  - prefix cache hit rate
  - mean queue, prefill and decode time per request (ms)
- **`results/quality.csv`:** one row per run. Columns:
  - run, date
  - A1 passes out of 20, with thinking off and with thinking on
  - A2 tasks resolved out of 50, A2 wall time, GPU-hours per resolved task
- **`docs/experiments.md`:** one entry per run, in this format. The skills are
  written from these entries ([skills plan](skills-plan.md)).

```
## r<NN>: <knob> <reference value> → <new value> (phase: prefill | decode)
- Reference run: r<MM>
- Why this knob: the gap from r<MM>'s analysis that it targets
- Prediction (and how it was made): …
- Startup facts: KV cache tokens, CUDA graph memory, start time, MoE kernel, attention backend
- Perf at c = C, vs the reference and the targets: TTFT p50/p90; E2E output tokens/s per user p50; ITL p50; output tokens/s; goodput
- Time split: queue / prefill / decode time per request; share of prompt tokens served from cache
- Probes: prefill tokens/s at c = 1 and 4; ITL at c = 1 … 128
- Quality: A1 x/20 and x/20; A2 x/50 (reference y/50)
- Explanation (the metric or profile that shows why): …
- Decision: keep or revert. New reference: r<NN> or r<MM>
- Next gap → next knob: …
- Rule learned: "use X when …, avoid when …"
```

## 6. Gap analysis

After each run, work through four steps. They pick the next knob.

1. **Find the biggest gap.** At C, how far is TTFT p50 from the stretch goal
   (0.26 s), and E2E output speed p50 from it (320 tokens/s)? If either misses
   the floor, that miss comes first.
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
| TTFT high, mostly queue time | Queue time ≫ prefill time; `num_requests_waiting` above 0 | Capacity: requests wait for a batch slot or KV space | D2, D5 |
| TTFT high, mostly prefill time, few cache hits | `prompt_tokens_cached` small relative to `prompt_tokens` | Prefill: the cache isn't saving work | P1; KV capacity (D4, D5) |
| TTFT high, mostly prefill time, many cache hits | `request_prefill_kv_computed_tokens` still large; prefill probe below the analytical model | Prefill: compute | P2 |
| ITL high even at concurrency 1–4 | Decode probe ITL at c=1 well above the analytical model | Decode: cost per step (bytes read, kernel launches) | D1, D3; CUDA graph coverage (D2) |
| ITL p50 fine, p99 spikes | Spikes line up with large steps in `iteration_tokens_total` | Prefill interrupting decode | P2 (smaller chunks) |
| ITL climbs steeply with concurrency | Decode probe ITL against batch size; `kv_cache_usage_perc` | Decode: batch growth | D2, D4 |
| E2E output speed low, ITL fine | TTFT is a large share of request latency | Prefill dominates the request | P1, P2 |

When two knobs fit the symptom equally, take the one listed first in its menu
below.

## Knob menus

Each knob is one flag added to `~/serve.sh` or removed from it
([getting-started step 8](getting-started.md#8-start-the-server)).

### Prefill

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| P1 | Prefix caching | on → off (`--no-enable-prefix-caching`) | How much of the agent's TTFT the cache is saving. Run it first, because it shows how much real prefill is left to optimize (Kiely §5.3.1, p. 136) |
| P2 | Chunked-prefill token budget | the default from the startup log → `--max-num-batched-tokens` 2k / 4k / 8k / 32k | TTFT vs ITL: large chunks finish prefills sooner but stall the decodes sharing the batch (Kiely §5.5). This knob affects decode too, so record both. It also sizes the activation memory vLLM reserves at startup, which comes out of the KV cache |

### Decode

| # | Variable | Baseline → variant | What to look for |
|---|---|---|---|
| D1 | Speculative decoding | off → DSpark, the recipe's `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'` → MTP, `'{"method":"mtp","num_speculative_tokens":3}'` | Acceptance on coding traffic (B4), speedup at C and at c1, and A3. Expect the gain to shrink as batches grow (Kiely §5.2.1). Each decode step runs batch × (1 + drafts) tokens, so check in the startup log that the CUDA graph capture sizes cover that, and raise `--max-cudagraph-capture-size` if not |
| D2 | Batch and CUDA graph sizes | default → `--max-num-seqs` 32 / 64 / 128; `--max-cudagraph-capture-size` ≥ batch × (1 + drafts) | Where per-user speed falls off as the batch grows, and whether decode steps fall outside the captured graphs |
| D3 | MoE kernel | DeepGEMM MegaMoE → vLLM's own choice (remove `--moe-backend deep_gemm_mega_moe` and `--enable-expert-parallel`) | What the fused MoE kernel buys per decode step. Record which kernel vLLM picks instead |
| D4 | KV cache dtype | fp8 → auto | What fp8 KV costs in accuracy (A2) and buys in KV capacity |
| D5 | GPU memory utilization | 0.9 (default) → `--gpu-memory-utilization 0.95` | More KV cache: fewer preemptions, and more of the prefix cache kept |

**The final configuration** combines whichever variants won, measured
together as one run. Combinations interact, so measure them; don't add up the
individual gains. Then run the full sweep on it, as for the baseline.

## Analytical model and profiles

- **Predict before the baseline:**
  - Decode step time at each of the decode probe's batch sizes, from bytes
    read per step ÷ HBM bandwidth (~8 TB/s). About 13B parameters are active
    per token: the routed experts in FP4 (half a byte per parameter), most
    other weights in FP8 (one byte). Work out the bytes from `config.json`,
    then add the KV cache reads.
  - Prefill time for the prefill probe, from FLOPs ÷ achievable compute.
  - The compute ceiling for the experts. Record which MoE kernel the startup
    log shows: its math rate sets the ceiling. Dense, per GPU: ~13.5 PFLOPS at
    FP4, ~4.5 at FP8, ~2.25 at BF16 ([ADR 0003](adr/0003-deepseek-v4-flash-0731-on-1xb300-with-vllm.md)).

  Then compare the predictions with the probes. The target is decode within
  about 25% at five or more batch sizes. That is a judgment call, as in ADR
  0001. Kiely §2.4 (pp. 61–64) shows how to prove which resource bounds each
  phase: compare an operation's arithmetic intensity with the GPU's ops:byte
  ratio.
- **Profile two configurations,** one decode-heavy (the decode probe at
  concurrency 64 or 128) and one prefill-heavy (the prefill probe at
  concurrency 1). Use vLLM's torch profiler: check `vllm serve --help` in
  v0.33 for the profiler flag, then call `/start_profile` and `/stop_profile`
  around a short run. AIPerf's `--server-profiler` flag can make those calls
  for you. Add Nsight Systems if it's available. Attribute the time to:
  - MoE GEMMs
  - attention and the sparse indexer
  - launch gaps and idle time

## Done when

- [ ] **Reproducible serving.** The image digest, model revision and launch
  command are in the repo. Someone else can bring the server up from
  getting-started alone.
- [ ] **Targets recorded.** The Artificial Analysis values used, with the date.
- [ ] **Baseline complete.** The analytical model's predictions are written
  down first. The sweep has zero failed requests, C is chosen against the
  targets, both probes have run, A1 passes, and the A2 resolve rate is
  recorded.
- [ ] **Every run went through the loop.** Each has a prediction, perf and
  quality results, a gap analysis and a decision in `docs/experiments.md`, and
  rows in `results/`.
- [ ] **Prefill.** P1 and P2 are done, with their effect on both TTFT and ITL.
- [ ] **Decode.** D1 and D2 are done, plus at least one of D3, D4 and D5.
- [ ] **Final configuration.** Measured at C against the baseline and the
  targets, then swept: the highest concurrency meeting the floor, baseline vs
  final.
- [ ] **Quality on the final configuration.** A1 passes; A2 shows no
  regression in the paired comparison with the baseline; A3 shows speculative
  decoding is lossless, or the divergences are explained.
- [ ] **Analytical model.** Predicted vs measured decode step time at five or
  more batch sizes, with every gap over 25% explained by a profile.
- [ ] **Two profiles** (decode-heavy and prefill-heavy) with time attribution.
- [ ] **Packaged endpoint.** The final configuration as a single launch script
  plus a README listing its targets, measured results, and known limits.
- [ ] **Write-up.** About two pages, with four charts:
  - TTFT p50 vs concurrency, with the floor and stretch lines
  - E2E output speed per user vs concurrency, with the same lines
  - total output tokens/s vs tokens/s per user (the trade-off curve)
  - speculative decoding speedup vs concurrency

**After the PoC:** turn the experiment log into two shareable Claude skills,
one for prefill and one for decode. See the [skills plan](skills-plan.md).

## Schedule

| Dates | Work |
|---|---|
| Sun 11-08 – Mon 11-09 | **Baseline.** Write the analytical model's predictions. Analyze the traces. Run the sweep, choose C, repeat C for noise. Run the probes, A1 and A2 |
| Tue 11-10 – Tue 11-17 | **Prefill and decode runs,** one knob per run at C, picked by each gap analysis. Profile the prefill-heavy and decode-heavy configurations |
| Wed 11-18 – Fri 11-20 | **Final configuration:** the loop, A3 and the full sweep. Package and write up |

If runs take longer than planned, cut D5, then D4, then D3, before cutting a
quality check.

## Keep out of the repo

Customer names, deal pricing, and anything quoted from internal channels.
