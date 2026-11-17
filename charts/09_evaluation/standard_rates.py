#!/usr/bin/env python3
"""Completeness at standard false-alarm rates, with the same criteria as evaluate_test_64b.py.

Reports, for one run:
  - FA at S/N >= 6 (noise false alarms per beam·hour);
  - threshold, completeness A / B / partial and FP in negatives at 10 FA per beam·hour;
  - the same at 10 FA per hour of the SYSTEM (10 / nbeams per beam·hour). If the signal-free time is not enough to
    resolve that rate, the first threshold without any noise event is used and a note is printed.

Usage: python 09_evaluation/standard_rates.py RUN_FOLDER [--log all_events.log]
       (sifter.log by default; with --log another log in the same format is evaluated, e.g. the grouper's full log)
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", type=Path)
ap.add_argument("--log", default="sifter.log")
a = ap.parse_args()

# reuse the evaluator's logic (event reading, signal windows, hits) on a temporary folder
D = a.run.resolve()
tmp = Path(tempfile.mkdtemp())
for f in ("assignment.csv", "asignacion.csv", "emitter.json", "emisor.json", "server.log", "gpu.csv", "vmstat.log",
          "config_dedispersion.yml"):
    if (D / f).exists():
        (tmp / f).symlink_to(D / f)
(tmp / "sifter.log").symlink_to(D / a.log)
src = (HERE / "evaluate_test_64b.py").read_text()
src = src[:src.index("def md_table")]
sys.argv = [str(HERE / "evaluate_test_64b.py"), str(tmp)]
g = {"__name__": "rates", "__file__": str(HERE / "evaluate_test_64b.py")}
exec(src, g)
shutil.rmtree(tmp)

th, fp, row, hours, nbeams = g["thresholds"], g["fp_curve"], g["row"], g["hours"], g["nbeams"]
print(f"# Standard rates · {D.name} ({a.log})\n")
print(f"Signal-free time: {hours:.1f} beam·hours (rate resolution: {1 / hours:.2f} FA per beam·hour)\n")
print(f"FA at S/N >= 6: {row(6.0)['noise FP/h']:.0f} per beam·hour\n")
print("| Target | S/N threshold | Measured FA (per beam·hour) | Completeness A | B | partial | FP in negatives |")
print("|---|---|---|---|---|---|---|")
for target, name in ((10.0, "10 per beam·hour"), (10.0 / nbeams, f"10/h of the {nbeams}-beam system")):
    k = np.flatnonzero(fp <= target)
    if not len(k):
        print(f"| {name} | not reached | | | | | |"); continue
    u = float(th[k[0]]); f = row(u)
    note = " (no noise events: bound)" if fp[k[0]] == 0 and target < 1 / hours else ""
    print(f"| {name} | {u:.2f} | {fp[k[0]]:.2f}{note} | {f['Recall A DM≤1000 (%)']} % | {f['Recall B DM≤1000 (%)']} % | "
          f"{f['Recall partial DM≤1000 (%)']} % | {f['non-FRB FP (% NEG DM≤1000)']} % |")
