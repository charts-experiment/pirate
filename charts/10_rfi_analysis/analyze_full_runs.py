#!/usr/bin/env python3
"""Full runs (4000 files) with mask m1: with the site RFI vs control without RFI.
Evaluates with ALL events (all_events.log, no caps) and with what reached the sifter (cap 4/beam), and corrects
for chance matches (shifted times).
Usage (pirate environment): export CHARTS_RFI_RESULTS=FOLDER; python analyze_full_runs.py
Runs: completa_m1_sin_rfi and completa_m1_rfi (folder names of the existing runs). Output: FOLDER/full_runs_m1.md
"""
import csv
import re

import numpy as np

from _common import R, RUNS_DIR, K_TOP, existing, all_events_log, evaluate, start_s

EV = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")


def chance(e, rows, threshold):
    e = e[e[:, 4] >= threshold]
    t = e[:, 2] / 200e3 - e[:, 1] * K_TOP
    order = np.argsort(e[:, 0], kind="stable"); e, t = e[order], t[order]
    first = {b: np.searchsorted(e[:, 0], b) for b in range(65)}

    def ok(r, off):
        b, dm = int(r["beam"]), float(r["dm"])
        a, z = first[b], first[b + 1]
        tt = start_s(r) + float(r["t_top_s"]) + off
        tol = max(0.05, float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000)
        return ((np.abs(e[a:z, 1] - dm) <= 0.1 * dm) & (np.abs(t[a:z] - tt) <= tol) & (e[a:z, 1] >= 20)).any()
    res = {}
    for s in ("A", "B", "P"):
        f = [r for r in rows if r["set"] == s and float(r["dm"]) <= 1000]
        real = np.mean([ok(r, 0) for r in f]); by_chance = np.mean([ok(r, o) for r in f for o in (-1.5, -1.0, 1.0, 1.5, 2.5, 3.5)])
        res[s] = (len(f), real, by_chance, 1 - (1 - real) / (1 - by_chance))
    return res


txt = ["# Full runs with mask m1: site RFI vs control without RFI\n",
       "4000 files in 64 beams (3510 s of stream per beam); grouper v1 `-t 6 --no-windows --max-cand 512 --log-all`; "
       "mask m1 (17.5 % of the band). Metrics with DM ≤ 1000. **Noise FP per beam·hour** (10/h of the 64-beam system ≈ 0.16 per beam·hour).\n"]
for name, desc in (("completa_m1_sin_rfi", "No RFI + m1 (control)"), ("completa_m1_rfi", "Site RFI + m1")):
    d = RUNS_DIR / name
    log_all = all_events_log(d)
    e = np.array([[float(x) for x in m.groups()] for m in EV.finditer(log_all.read_text(errors="replace"))]).reshape(-1, 5)
    rows = list(csv.DictReader(open(existing(d, "assignment.csv", "asignacion.csv"))))
    nch = 1714
    ich = (e[:, 2] // (2048 * 200)).astype(int)
    pbc = np.bincount(ich * 64 + e[:, 0].astype(int), minlength=nch * 64)
    txt += [f"\n## {desc} (`{name}`)\n",
            f"- Events (all, S/N > 6): {len(e)}; per beam and chunk: mean {pbc.mean():.2f}, p99 {np.percentile(pbc, 99):.0f}, max {pbc.max()}; "
            f"beam-chunks with > 4 events: {(pbc > 4).mean() * 100:.2f} %."]
    for thr in (6.0, 7.0):
        ch = chance(e, rows, thr)
        txt.append(f"- Chance matches with S/N ≥ {thr:g} (A / B / cut): " +
                   " / ".join(f"{ch[s][2] * 100:.1f} %" for s in ("A", "B", "P")) +
                   "; corrected real detection: " + " / ".join(f"{ch[s][3] * 100:.1f} %" for s in ("A", "B", "P")) +
                   f" (n = {ch['A'][0]} / {ch['B'][0]} / {ch['P'][0]}).")
    for label, log, t in (("all", log_all, "all events (no cap)"), ("capped", d / "sifter.log", "what reached the sifter (cap 4 per beam)")):
        md, _ = evaluate(d, log, label)
        md_rows = [l for l in md.splitlines() if re.match(r"^\| [0-9]", l)]
        txt += [f"\n**With {t}:**\n", "| S/N threshold | noise FP/h | Recall A | Recall B | Recall cut | non-FRB FP % | A DM≤1105 | B DM≤1105 |",
                "|---|---|---|---|---|---|---|---|"] + md_rows[:4]
        if len(md_rows) > 4:
            txt += ["\n| target FP/h | threshold | FP/h | A | B | cut | non-FRB % | A≤1105 | B≤1105 |", "|---|---|---|---|---|---|---|---|---|"] + md_rows[4:]
    perf_prefixes = ("| Emitter rate", "| Server capacity", "| GPU", "| Ritmo", "| Capacidad")
    perf = [l for l in existing(d, "results.md", "resultados.md").read_text().splitlines() if l.startswith(perf_prefixes)]
    txt += ["\n**Performance:**\n", "| | |", "|---|---|"] + perf
(R / "full_runs_m1.md").write_text("\n".join(txt) + "\n")
print("\n".join(txt))
