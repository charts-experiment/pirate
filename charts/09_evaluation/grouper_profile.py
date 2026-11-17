#!/usr/bin/env python3
"""Detection time per block from the grouper v1 profile (option --profile → grouper_timings.csv).

Separates, per 2.048 s block, the time the grouper spends WAITING for pirate from the time it spends WORKING, and
breaks down the stages. If the waiting is large, the bottleneck is upstream (emitter or reception); the detection's
own time is pirate (time_dedisperser) + the grouper working.

Usage: python 09_evaluation/grouper_profile.py RUN_FOLDER [--skip 5]
"""
import argparse
import csv
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", type=Path)
ap.add_argument("--skip", type=int, default=5, help="start-up blocks that are not counted")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.run / "grouper_timings.csv")))[a.skip:-2]
tot = [float(x["total_ms"]) for x in rows]
wait = [float(x["wait_pirate_ms"]) for x in rows]
work = sorted(t - e for t, e in zip(tot, wait))
print(f"# Grouper profile · {a.run.name} ({len(rows)} blocks)\n")
print("| | median (ms) |\n|---|---|")
print(f"| Period per block | {st.median(tot):.0f} |")
print(f"| Waiting for pirate | {st.median(wait):.0f} |")
print(f"| Working | {st.median(work):.0f} (p90 {work[int(0.9 * len(work))]:.0f}) |")
stages = [k[:-3] for k in rows[0] if k.endswith("_ms") and k not in ("total_ms", "wait_pirate_ms")]
for e in stages:
    v = [float(x[e + "_ms"]) for x in rows]
    if st.median(v) >= 0.5:
        print(f"| · {e} | {st.median(v):.1f} |")
print(f"\nDecoded candidates per block: median {st.median(float(x['n_cand']) for x in rows):.0f}; "
      f"v1 events: {st.median(float(x['n_v1']) for x in rows):.0f}")
