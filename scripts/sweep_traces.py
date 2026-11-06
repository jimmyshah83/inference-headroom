#!/usr/bin/env python3
"""Run the coding-agent trace sweep (docs/poc.md, step 3: Perf run).

Replays SemiAnalysis's Claude Code traces under InferenceX's AgentX scenario.
Each concurrency gets its own AIPerf run, because the scenario refuses a sweep
in one command. The workload flags below are fixed: every run in the PoC uses
the same ones, so a difference in the results comes from the server. Only the
run name, the concurrencies, the repeats and the URL can change.

Examples (from the repo root, with the .venv from `uv sync`):

    # Print the AIPerf commands without running them.
    python scripts/sweep_traces.py r00-baseline --dry-run

    # Test one point first.
    python scripts/sweep_traces.py r00-baseline --concurrency 1

    # Run 0: three repeats of the full sweep. On the Mac, keep it awake.
    caffeinate -i python scripts/sweep_traces.py r00-baseline --repeats 3

    # A later run: one repeat.
    python scripts/sweep_traces.py r01-o6-prefix-cache-off

Each point writes its AIPerf artifacts to
<runs-dir>/<run>/traces/rep<r>/c<c>/, and the sweep appends one line per point
to <runs-dir>/<run>/traces/progress.tsv. <runs-dir> is the repo's runs/ folder
unless --runs-dir says otherwise. A point that already has its summary
(profile_export_aiperf.json) is skipped, so rerunning the same command resumes
an interrupted sweep. Ctrl-C stops the whole sweep.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The fixed workload. Changing any of these means re-running the baseline.
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DATASET = "semianalysis_cc_traces_weka_062126_256k"
SCENARIO = "inferencex-agentx-mvp"
CACHE_BUST = "first_turn_prefix"
AGENTIC_WARMUP_S = 120
RANDOM_SEED = 42
DURATION_S = 900  # The scenario requires at least 900 s per point.
GRACE_S = 300
GOODPUT = "time_to_first_token:3000 inter_token_latency:20"

DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32]  # InferenceX's task concurrencies.
SUMMARY_FILE = "profile_export_aiperf.json"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"  # The repo's runs/ folder.


def aiperf_binary() -> str:
    """Find aiperf next to this Python (the venv) or on PATH."""
    beside = Path(sys.executable).parent / "aiperf"
    if beside.exists():
        return str(beside)
    found = shutil.which("aiperf")
    if found:
        return found
    sys.exit("aiperf not found. Run `uv sync` and activate .venv, then try again.")


def aiperf_command(aiperf: str, url: str, concurrency: int, artifact_dir: Path) -> list[str]:
    """Build the AIPerf command for one point of the sweep."""
    return [
        aiperf, "profile",
        "--model", MODEL,
        "--url", url, "--endpoint-type", "chat", "--streaming",
        "--tokenizer", MODEL, "--tokenizer-revision", MODEL_REVISION,
        "--scenario", SCENARIO, "--cache-bust", CACHE_BUST,
        "--public-dataset", DATASET,
        "--agentic-cache-warmup-duration", str(AGENTIC_WARMUP_S),
        "--random-seed", str(RANDOM_SEED),
        "--concurrency", str(concurrency),
        "--benchmark-duration", str(DURATION_S),
        "--benchmark-grace-period", str(GRACE_S),
        "--goodput", GOODPUT,
        "--artifact-dir", str(artifact_dir),
    ]


def check_health(url: str) -> None:
    """Stop early if the replica (or the SSH tunnel to it) isn't reachable."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10) as resp:
            if resp.status == 200:
                return
            reason = f"HTTP {resp.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = str(exc)
    sys.exit(f"{url}/health failed ({reason}). Is the replica up and the tunnel open?")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the coding-agent trace sweep with AIPerf (docs/poc.md, step 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run", help="run name, e.g. r00-baseline or r01-o6-prefix-cache-off")
    parser.add_argument(
        "--concurrency", default=",".join(map(str, DEFAULT_CONCURRENCIES)),
        help="comma-separated concurrencies (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="passes of the sweep; run 0 uses 3 to measure noise (default: %(default)s)",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="replica to measure: replica A, or the router in O8 (default: %(default)s)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS_DIR,
        help="where run folders go (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    traces_dir = args.runs_dir.expanduser() / args.run / "traces"
    aiperf = aiperf_binary() if not args.dry_run else "aiperf"

    points = [(r, c) for r in range(1, args.repeats + 1) for c in concurrencies]
    if args.dry_run:
        for r, c in points:
            print(shlex.join(aiperf_command(aiperf, args.url, c, traces_dir / f"rep{r}" / f"c{c}")))
        return 0

    check_health(args.url)
    traces_dir.mkdir(parents=True, exist_ok=True)
    progress = traces_dir / "progress.tsv"
    if not progress.exists():
        progress.write_text("start_utc\tend_utc\trepeat\tconcurrency\texit_code\n")

    failed: list[tuple[int, int, int]] = []
    for i, (r, c) in enumerate(points, start=1):
        point_dir = traces_dir / f"rep{r}" / f"c{c}"
        label = f"[{i}/{len(points)}] rep {r}, concurrency {c}"
        if (point_dir / SUMMARY_FILE).exists():
            print(f"{label}: already done, skipping ({point_dir})")
            continue

        print(f"\n{label}: starting at {now()}, writing to {point_dir}", flush=True)
        start = now()
        try:
            code = subprocess.run(aiperf_command(aiperf, args.url, c, point_dir)).returncode
        except KeyboardInterrupt:
            with progress.open("a") as f:
                f.write(f"{start}\t{now()}\t{r}\t{c}\tinterrupted\n")
            print(f"\nStopped during {label}. Rerun the same command to resume.")
            return 130
        with progress.open("a") as f:
            f.write(f"{start}\t{now()}\t{r}\t{c}\t{code}\n")
        if code != 0:
            failed.append((r, c, code))
            print(f"{label}: AIPerf exited with {code}. See {point_dir / 'logs' / 'aiperf.log'}")

    if failed:
        print("\nFailed points (rerun the same command to retry them):")
        for r, c, code in failed:
            print(f"  rep {r}, concurrency {c}: exit {code}")
        return 1
    print(f"\nSweep complete: {traces_dir}")
    return 0


if __name__ == "__main__":
    # Let Ctrl-C reach both AIPerf and this script, and stop the whole sweep.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
