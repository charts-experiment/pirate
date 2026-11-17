#!/usr/bin/env python3
"""Diagnosis of grouper v1 grouping with the site RFI: compares short runs with RFI that change one grouper
parameter, using ALL events (all_events.log, without the --k/--k-total caps).

  corta_todos_rfi      reference (tol-t 0.1 s, cell threshold 5)
  diag_tol             --tol-t 0.002 (almost no discarding by t_top)
  diag_celda           --cell-threshold 7 (regions only with cells of S/N ≥ 7)
  corta_todos_sin_rfi  control without RFI
  mask_m{1,2}_{rfi,sin_rfi}  with the static channel masks
(folder names of the existing runs)

For each: total events, per beam and chunk, Recall/FP (evaluate_test_64b.py on <run>/all/), chance matches and the
best S/N found for each A, B and cut FRB with DM ≤ 1000.
Usage (pirate environment): export CHARTS_RFI_RESULTS=FOLDER; python compare_diag.py [OUTPUT.md]
Output: FOLDER/grouping_diagnosis.md
"""
import csv
import re
import sys

import numpy as np

from _common import R, RUNS_DIR, K_TOP, existing, all_events_log, evaluate, start_s

RUNS = [("corta_todos_sin_rfi", "no RFI (control)"), ("corta_todos_rfi", "RFI, reference"),
        ("diag_tol", "RFI, --tol-t 0.002"), ("diag_celda", "RFI, --cell-threshold 7"),
        ("mask_m1_sin_rfi", "no RFI + mask m1"), ("mask_m2_sin_rfi", "no RFI + mask m2"),
        ("mask_m1_rfi", "RFI + mask m1"), ("mask_m2_rfi", "RFI + mask m2")]
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "grouping_diagnosis.md"
EV = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+)")
SEQ, T = 200, 2048


def chance(e, rows):
    """Matches at the true time and by chance (shifted times), with the evaluator's criteria."""
    t = e[:, 2] / 200e3 - e[:, 1] * K_TOP

    def ok(r, off):
        b, dm = int(r["beam"]), float(r["dm"])
        tt = start_s(r) + float(r["t_top_s"]) + off
        tol = max(0.05, float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000)
        return ((e[:, 0] == b) & (np.abs(e[:, 1] - dm) <= 0.1 * dm) & (np.abs(t - tt) <= tol) & (e[:, 1] >= 20)).any()
    f = [r for r in rows if r["set"] in ("A", "B", "P") and float(r["dm"]) <= 1000]
    real = np.mean([ok(r, 0) for r in f]); by_chance = np.mean([ok(r, o) for r in f for o in (-1.5, -1.0, 1.0, 1.5, 2.5, 3.5)])
    return real, by_chance, 1 - (1 - real) / (1 - by_chance)


txt = ["# Diagnosis of grouper v1 grouping with the site RFI\n",
       "All events with S/N > 6, no caps; 80 files (DM ≤ 1000: 6 per set). Noise FP per beam·hour.\n",
       "| Run | Total events | Per beam and chunk (median) | S/N threshold 8: noise FP/h | Recall A | Recall B | Recall cut | non-FRB FP % | Matches A+B+cut | By chance | Corrected real detection |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
bests, rows = {}, None
for name, desc in RUNS:
    d = RUNS_DIR / name
    log = all_events_log(d)
    if log is None:
        txt.append(f"| {desc} | (missing) | | | | | | | | | |")
        continue
    e = np.array([[float(x) for x in m.groups()] for m in EV.finditer(log.read_text(errors="replace"))]).reshape(-1, 3)
    if len(e) == 0:
        txt.append(f"| {desc} | 0 | 0 | | | | | | | | |")
        continue
    ich = (e[:, 2] // (T * SEQ)).astype(int)
    pbc = np.unique(ich * 1000 + e[:, 0].astype(int), return_counts=True)[1]
    md, best = evaluate(d, log, "all")
    bests[name] = best
    run_rows = list(csv.DictReader(open(existing(d, "assignment.csv", "asignacion.csv"))))
    rows = rows or run_rows
    r8 = [l for l in md.splitlines() if l.startswith("| 8.0 |")]
    c = r8[0].split("|")[1:-1] if r8 else ["8.0"] + ["—"] * 7
    re_, ch_, co_ = chance(e, run_rows)
    txt.append(f"| {desc} | {len(e)} | {np.median(pbc):.0f} | {c[1].strip()} | {c[2].strip()} | {c[3].strip()} | {c[4].strip()} | {c[5].strip()} | {re_*100:.0f} % | {ch_*100:.0f} % | {co_*100:.0f} % |")

txt += ["\n## Best S/N found for each FRB (DM ≤ 1000; empty = not detected)\n",
        "| File | DM | Expected S/N | " + " | ".join(desc for name, desc in RUNS if name in bests) + " |",
        "|---|---|---|" + "---|" * len(bests)]
for i, r in enumerate(rows or []):
    if r["set"] not in ("A", "B", "P") or float(r["dm"]) > 1000:
        continue
    exp = r.get("snr_esperado") or r.get("snr_visible_esperado") or ("8" if r["set"] == "A" else "")
    vals = []
    for name, desc in RUNS:
        if name in bests:
            v = bests[name][i]
            vals.append(f"{v:.1f}" if np.isfinite(v) else "")
    txt.append(f"| {r['name']} | {float(r['dm']):.0f} | {float(exp):.1f} | " + " | ".join(vals) + " |")
(R / OUTPUT).write_text("\n".join(txt) + "\n")
print("\n".join(txt))
