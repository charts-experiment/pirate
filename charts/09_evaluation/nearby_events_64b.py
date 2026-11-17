#!/usr/bin/env python3
"""For each positive file of a 64-beam test, the nearest pirate events (to classify losses).

Uses the same sifter.log parsing and the same time conversion as evaluate_test_64b.py. Per file it stores:
  snr_dm_ok, dm_dm_ok        best event in the time window with DM ±10 % (any S/N ≥ 5 delivered by the grouper)
  snr_other_dm, dm_other_dm  best event in the time window with DM outside ±10 %
Time window: |Δt| ≤ max(0.1 s, burst duration) around the expected arrival at 501.6 MHz.

Usage: python nearby_events_64b.py RUN
Output: RUN/analysis_by_parameter/nearby_events.csv (read by loss_analysis_64b.py)
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = Path(sys.argv[1]).resolve()
src = (HERE / "evaluate_test_64b.py").read_text()
src = src[:src.index("# best S/N of an event")]            # event and ground-truth parsing, without evaluating
sys.argv = [str(HERE / "evaluate_test_64b.py"), str(D)]
g = {"__file__": str(HERE / "evaluate_test_64b.py"), "__name__": "nearby_events"}
exec(compile(src, "evaluate_test_64b.py", "exec"), g)
ev_beam, ev_dm, ev_snr, ev_t, rows, TOL_DM = (g[k] for k in ("ev_beam", "ev_dm", "ev_snr", "ev_t", "rows", "TOL_DM"))

OUT = D / "analysis_by_parameter"
OUT.mkdir(exist_ok=True)
with open(OUT / "nearby_events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "snr_dm_ok", "dm_dm_ok", "snr_other_dm", "dm_other_dm"])
    for r in rows:
        if r["set"] == "NEG":
            continue
        ok_t = (ev_beam == r["b"]) & (np.abs(ev_t - r["t_abs"]) <= max(0.1, r["box"]))
        ok_dm = np.abs(ev_dm - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0)
        row = [r["name"]]
        for m in (ok_t & ok_dm, ok_t & ~ok_dm):
            if m.any():
                k = np.argmax(np.where(m, ev_snr, -np.inf))
                row += [f"{ev_snr[k]:.2f}", f"{ev_dm[k]:.1f}"]
            else:
                row += ["", ""]
        w.writerow(row)
print("done", OUT / "nearby_events.csv")
