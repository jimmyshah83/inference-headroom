#!/usr/bin/env python3
"""Combine a trace sweep's repeats into one table (docs/poc.md, 3. Perf run, step 11).

AIPerf summarizes each point on its own. sweep_traces.py runs one `aiperf
profile` per point, so nothing combines rep1, rep2 and rep3. For each
concurrency and metric, this script reports:

- **Each repeat's value, their mean, and the spread.** The spread is max - min
  across the repeats, also given as a % of the mean. Run 0's spread is the
  noise floor. If a later run differs from its reference by less than twice
  that spread, repeat it before you believe it.
- **Pooled latency percentiles.** For TTFT, ITL and request latency, it pools
  the per-request records of all the repeats and takes the percentile of the
  pool. The mean of three p90s isn't the p90 of all their requests, so use
  the pooled value as the headline number and the per-repeat values to judge
  the noise.

It reads only finished output, so run it on the Mac after copying the traces
(step 10). It uses only the standard library, so it also runs on the node.

    python scripts/aggregate_sweep.py r00-baseline
    python scripts/aggregate_sweep.py r00-baseline --runs-dir ~/runs   # on the node

It prints a table and writes <runs-dir>/<run>/traces/aggregate.csv, with one
row per concurrency and metric. It exits 1 if a point was cancelled, is
incomplete or had failed requests, because any error invalidates the point.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"  # The repo's runs/ folder.
SUMMARY_FILE = "profile_export_aiperf.json"
RECORDS_FILE = "profile_export.jsonl"
OUTPUT_FILE = "aggregate.csv"
POINT_DIR = re.compile(r"^rep(\d+)/c(\d+)$")

# (CSV name, unit, AIPerf metric, statistic). Percentiles are also pooled.
METRICS = [
    *(
        (f"{short}_p{p}", "ms", metric, f"p{p}")
        for short, metric in (
            ("ttft", "time_to_first_token"),
            ("itl", "inter_token_latency"),
            ("request_latency", "request_latency"),
        )
        for p in (50, 90, 99)
    ),
    ("output_tok_s", "tokens/s", "output_token_throughput", "avg"),
    ("goodput_req_s", "requests/s", "goodput", "avg"),
    ("requests", "requests", "request_count", "avg"),
]
POOLED_METRICS = sorted({metric for _, _, metric, stat in METRICS if stat.startswith("p")})

# The printed table: (column header, CSV name). Latency columns show the pooled
# value, the others the mean; each is followed by the spread.
TABLE = [
    ("TTFT p90 ms", "ttft_p90"),
    ("ITL p90 ms", "itl_p90"),
    ("Latency p90 ms", "request_latency_p90"),
    ("Output tok/s", "output_tok_s"),
    ("Goodput req/s", "goodput_req_s"),
]


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between the closest ranks, as numpy and AIPerf do."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_point(point_dir: Path) -> dict:
    """Read one point's summary and, if present, its per-request records."""
    summary = json.loads((point_dir / SUMMARY_FILE).read_text())
    values = {name: summary.get(metric, {}).get(stat) for name, _, metric, stat in METRICS}

    problems = []
    if summary.get("was_cancelled"):
        problems.append("cancelled")
    if not summary.get("is_complete", True):
        problems.append("incomplete")
    if summary.get("error_summary"):
        problems.append(f"{len(summary['error_summary'])} error type(s) in the summary")
    if not summary.get("metadata", {}).get("submission_valid", True):
        problems.append("submission_valid is false")

    # Only the profiling phase counts: the file also holds the warmup's requests.
    samples: dict[str, list[float]] | None = None
    failed = None
    records = point_dir / RECORDS_FILE
    if records.exists():
        samples = {metric: [] for metric in POOLED_METRICS}
        failed = 0
        with records.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("metadata", {}).get("benchmark_phase") != "profiling":
                    continue
                if rec.get("error"):
                    failed += 1
                    continue
                metrics = rec.get("metrics", {})
                for metric in POOLED_METRICS:
                    if metric in metrics:
                        samples[metric].append(metrics[metric]["value"])
        if failed:
            problems.append(f"{failed} failed request(s)")
    values["failed_requests"] = failed
    return {"values": values, "samples": samples, "problems": problems}


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine a trace sweep's repeats: mean, spread and pooled percentiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run", help="run name, e.g. r00-baseline")
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS_DIR,
        help="where run folders are (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traces_dir = args.runs_dir.expanduser() / args.run / "traces"
    if not traces_dir.is_dir():
        sys.exit(f"{traces_dir} doesn't exist. Copy the traces first (docs/poc.md, step 10).")

    # points[concurrency][repeat] = the loaded point; missing = folders without a summary.
    points: dict[int, dict[int, dict]] = {}
    missing: list[str] = []
    for point_dir in sorted(traces_dir.glob("rep*/c*")):
        match = POINT_DIR.match(point_dir.relative_to(traces_dir).as_posix())
        if not match or not point_dir.is_dir():
            continue
        rep, conc = int(match[1]), int(match[2])
        if not (point_dir / SUMMARY_FILE).exists():
            missing.append(f"rep{rep}/c{conc}")
            continue
        points.setdefault(conc, {})[rep] = load_point(point_dir)
    if not points:
        sys.exit(f"No finished points under {traces_dir}.")

    repeats = sorted({rep for reps in points.values() for rep in reps})
    rows = []
    for conc in sorted(points):
        reps = points[conc]
        for name, unit, metric, stat in [*METRICS, ("failed_requests", "requests", None, "sum")]:
            per_rep = [reps[r]["values"][name] if r in reps else None for r in repeats]
            present = [v for v in per_rep if v is not None]
            mean = sum(present) / len(present) if present else None
            spread = max(present) - min(present) if len(present) > 1 else None
            spread_pct = spread / mean * 100 if spread is not None and mean else None
            pooled = None
            if stat.startswith("p") and all(reps[r]["samples"] is not None for r in reps):
                pool = [v for r in reps for v in reps[r]["samples"][metric]]
                pooled = percentile(pool, float(stat[1:]))
            rows.append({
                "run": args.run, "concurrency": conc, "metric": name, "unit": unit,
                "repeats": len(present),
                **{f"rep{r}": v for r, v in zip(repeats, per_rep)},
                "mean": mean, "spread": spread, "spread_pct": spread_pct, "pooled": pooled,
            })

    out = traces_dir / OUTPUT_FILE
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # The table: one line per concurrency.
    by_key = {(row["concurrency"], row["metric"]): row for row in rows}
    print(f"{args.run}: repeats {', '.join(map(str, repeats))}; "
          f"concurrencies {', '.join(map(str, sorted(points)))}")
    print("Each cell: value (spread = max - min across repeats, as % of the mean).")
    print("Latency columns are pooled across the repeats; the others are means.\n")
    header = ["c", "reps", *(title for title, _ in TABLE), "Failed"]
    lines = [header]
    for conc in sorted(points):
        cells = [str(conc), str(len(points[conc]))]
        for _, name in TABLE:
            row = by_key[(conc, name)]
            value = row["pooled"] if row["pooled"] is not None else row["mean"]
            pct = f" ({row['spread_pct']:.1f}%)" if row["spread_pct"] is not None else ""
            cells.append(fmt(value) + pct)
        failed = [p["values"]["failed_requests"] for p in points[conc].values()]
        cells.append("?" if None in failed else str(sum(failed)))
        lines.append(cells)
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    for line in lines:
        print("  ".join(cell.rjust(w) for cell, w in zip(line, widths)))

    problems = [
        f"rep{rep}/c{conc}: {', '.join(p['problems'])}"
        for conc in sorted(points) for rep, p in sorted(points[conc].items()) if p["problems"]
    ]
    if missing:
        print(f"\nNot finished (no summary yet), left out: {', '.join(missing)}")
    if problems:
        print("\nInvalid points (read their logs/aiperf.log; docs/poc.md, step 9):")
        for problem in problems:
            print(f"  {problem}")
    print(f"\nWrote {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
