# Plot a trace sweep

AIPerf 0.13.0, the version pinned in `pyproject.toml`, has an `aiperf plot`
command. It reads the sweep's output folders as they are and draws charts from
them, so nothing else needs installing. It picks up the model name and each
point's concurrency from the files AIPerf already wrote.

Run it on the Mac after copying the traces and combining the repeats
([PoC plan](poc.md), 3. Perf run, steps 10 and 11). It reads only the copied
output and sends nothing to the node.

## Draw the charts

**Mac, repo root: plot the whole sweep.** A folder that holds several runs gets
comparison charts. `--output` keeps the PNGs with the run's other output, which
Git ignores.

```bash
. .venv/bin/activate
aiperf plot runs/r00-baseline/traces --config scripts/plot_config.yaml --output runs/r00-baseline/plots
```

**Mac, repo root: plot one point.** A single `c*` folder gets charts of that
point over time.

```bash
aiperf plot runs/r00-baseline/traces/rep3/c32 --config scripts/plot_config.yaml --output runs/r00-baseline/plots/rep3-c32
```

**Mac, repo root: explore them instead.** `--dashboard` starts an interactive
page at http://127.0.0.1:8050 instead of writing PNGs. Not tried yet.

```bash
aiperf plot runs/r00-baseline/traces --config scripts/plot_config.yaml --dashboard
```

Without `--config`, AIPerf writes its own default config to
`~/.aiperf/plot_config.yaml` and uses that. See the next section for why this
repo's config is better for the sweep.

## Why the repo has its own config

AIPerf's default config draws only 1 of its 4 comparison charts for this sweep:

- **Three charts need throughput per GPU.** AIPerf works that out from GPU
  telemetry, and the sweep doesn't collect any: each point's `telemetry_data`
  lists no endpoints. [`scripts/plot_config.yaml`](../scripts/plot_config.yaml)
  plots total output throughput instead. On one 8-GPU node, per GPU would be
  total ÷ 8, so the shapes are the same.
- **The fourth chart zigzags.** It puts TTFT on x, and AIPerf joins the points
  in x order. TTFT doesn't rise steadily with concurrency here, so the line
  jumps back and forth. The repo's config puts throughput on x, which does
  rise with concurrency, so each line follows the sweep.

## What each chart shows

From a whole sweep (all 18 points):

| File | Chart |
|---|---|
| `ttft_vs_throughput.png` | p90 TTFT against output tokens/s, labelled by concurrency |
| `itl_vs_throughput.png` | p50 ITL against output tokens/s |
| `interactivity_vs_throughput.png` | Total output tokens/s against tokens/s per user. The line joins the Pareto frontier; a point below it is worse on both axes than a point on it |
| `latency_vs_throughput_uncertainty.png` | Request latency against output tokens/s. It pools the 3 repeats of each concurrency into one mean with a 95% ellipse |

From one point:

| File | Chart |
|---|---|
| `ttft_over_time.png` | TTFT of each request, in order |
| `latency_over_time.png` | Request latency over the run, with rolling p50, p95 and p99 |
| `ttft_timeline.png` | Each request as a bar, split into time to first token and generation |

## What run 0's charts show (2026-11-08)

**c4 sits below the Pareto frontier in all three repeats.** It reaches about
the same output throughput as c16, but each user gets fewer tokens/s and ITL
is higher. Its requests are heavier, so this isn't the server getting worse at
c4: their average input is 167K–174K tokens, against 80K–131K at the other
concurrencies.

| Concurrency | Avg input tokens (rep1/2/3) | Avg output tokens (rep1/2/3) | Output tokens/s | Tokens/s per user | p50 ITL (ms) |
|---|---|---|---|---|---|
| 1 | 80K / 80K / 77K | 849 / 879 / 935 | 129 | 256 | 3.85 |
| 2 | 117K / 110K / 113K | 1181 / 890 / 973 | 221 | 245 | 4.03 |
| 4 | 172K / 167K / 174K | 1030 / 1003 / 1052 | 472 | 180 | 5.90 |
| 8 | 95K / 104K / 119K | 674 / 701 / 695 | 358 | 205 | 4.95 |
| 16 | 131K / 129K / 130K | 911 / 864 / 880 | 482 | 188 | 5.42 |
| 32 | 115K / 114K / 112K | 1028 / 1008 / 990 | 1131 | 138 | 7.20 |

The last three columns are means of the 3 repeats.

The requests differ from one concurrency to the next, so a step between two
concurrencies mixes the server's behavior with the change in requests. Check
the average input and output tokens before reading a bend in a curve as a
server effect.

## Limits

- **It doesn't read `aggregate.csv`.** AIPerf pools the repeats itself in the
  ellipse chart. With only 3 repeats, the ellipses are wide.
- **No line per repeat, and no line per run.** AIPerf names each point after
  its `c*` folder, so it can't draw `rep1`, `rep2` and `rep3`, or `r00` and
  `r01`, as separate curves. Comparing two runs will need another way, or a
  change to the folder layout.
- **PNGs need Chrome.** AIPerf renders them through Chrome, which the Mac has.
  That's another reason to plot on the Mac and not on the node.
- **Charts per GPU, and GPU utilization over time, need GPU telemetry.** That
  means running NVIDIA's DCGM exporter on the node during the sweep, which is
  an install there. Not decided yet.
