#!/usr/bin/env python3
"""Verifies the alerts of a run against the FRBs injected by the emitter.

For each alert in <run>/alerts.jsonl it looks for the file in the same beam whose arrival at 501.6 MHz matches
(same criteria as evaluate_test_64b.py: |ΔDM| ≤ 10 % and |Δt| ≤ max(50 ms, optimal boxcar); negatives ±100 ms).
Each alert is classified as true FRB, non-FRB (negative) or false alarm, and the FRBs that should have produced
an alert (expected S/N ≥ threshold and DM in range) but did not are listed.

Writes <run>/alerts.md. Usage: python 09_evaluation/verify_alerts.py RUN_FOLDER
"""
import csv
import json
import sys
from pathlib import Path

import yaml

TOL_DM, TOL_T, TOL_NEG = 0.10, 0.05, 0.10

if len(sys.argv) < 2:
    sys.exit(__doc__)
D = Path(sys.argv[1])
if not (D / "alerts.jsonl").exists():
    sys.exit(f"ERROR: {D / 'alerts.jsonl'} does not exist (did the run use 08_server/sifter/alert_sifter.py?)")
alerts = [json.loads(x) for x in (D / "alerts.jsonl").read_text().splitlines() if x.strip()]
cfg = yaml.safe_load(open(D / "test.yml")) if (D / "test.yml").exists() else {}
rule = (cfg.get("server") or {}).get("alerts") or {}
threshold, dm_min, dm_max = rule.get("threshold", 7.6), rule.get("dm_min", 20), rule.get("dm_max")
dm_search = 1105.0

rows = list(csv.DictReader(open(D / "assignment.csv")))
for r in rows:
    r["b"], r["t_abs"] = int(r["beam"]), int(r["start_sample"]) / 1000 + float(r["t_top_s"])
    r["dmv"] = float(r["dm_equivalente_pos"] if r["set"] == "NEG" else r["dm"])
    r["box"] = float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000
    r["snr_exp"] = float(r.get("snr_esperado") or r.get("snr_visible_esperado") or "nan")


def matches(al, r, beams=None):
    if r["b"] not in (beams or [al["beam"]]):
        return False
    dt = abs(al["t_stream_s"] - r["t_abs"])
    if r["set"] == "NEG":
        return dt <= TOL_NEG
    return abs(al["dm"] - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0) and dt <= max(TOL_T, r["box"])


alerts = [al for al in alerts if al.get("type", "alert") == "alert"]        # updates are not counted separately
alert_rows, detected = [], set()
for al in alerts:
    r = next((r for r in rows if matches(al, r)), None)
    kind = "false alarm (noise)" if r is None else ("non-FRB (negative)" if r["set"] == "NEG" else "**true FRB**")
    # the same pulse in several beams at once gives a single alert (coincident_beams): it counts for all of them
    others = [x["name"] for x in rows if x["set"] != "NEG" and matches(al, x, al.get("coincident_beams"))]
    detected.update(others)
    if r is None and others:
        r, kind = next(x for x in rows if x["name"] == others[0]), "**true FRB** (coincident beam)"
    alert_rows.append((al, r, kind))

expected = [r for r in rows if r["set"] != "NEG" and r["snr_exp"] >= threshold and r["dmv"] >= dm_min
            and r["dmv"] <= min(dm_search, dm_max or 1e9)]
missed = [r for r in expected if r["name"] not in detected]
lat = sorted(al["latency_s"] for al in alerts)

o = [f"# FRB alerts — {D.name}", "",
     f"Rule: S/N ≥ {threshold}, DM {dm_min}–{dm_max or dm_search:g}, veto if > {rule.get('max_beams', 8)} beams at once.", "",
     f"- Alerts: **{len(alerts)}** → true FRBs {sum(t.startswith('**true') for _, _, t in alert_rows)}, "
     f"non-FRB {sum(t.startswith('non-FRB') for _, _, t in alert_rows)}, false alarms {sum(t.startswith('false') for _, _, t in alert_rows)}",
     f"- Injected FRBs that should give an alert (expected S/N ≥ {threshold}, DM in range): {len(expected)}; "
     f"with alert {len(expected) - len(missed)}; without alert {len(missed)}"]
if lat:
    o.append(f"- Latency (end of the pulse at the bottom edge → alert): median {lat[len(lat) // 2]:.2f} s, "
             f"min {lat[0]:.2f} s, max {lat[-1]:.2f} s")
o += ["", "| alert | beam | DM | S/N | arrival (UTC, 501.6 MHz) | latency (s) | result | file | injected DM | expected S/N |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for al, r, kind in alert_rows:
    fname, dmi, snre = (r["name"], f"{r['dmv']:.1f}", f"{r['snr_exp']:.1f}") if r else ("—", "—", "—")
    o.append(f"| {al['n']} | {al['beam']} | {al['dm']} | {al['snr']} | {al['utc_arrival']} | {al['latency_s']} | {kind} | "
             f"{fname} | {dmi} | {snre} |")
if missed:
    o += ["", "FRBs without alert:", "", "| file | beam | DM | expected S/N |", "|---|---|---|---|"]
    o += [f"| {r['name']} | {r['b']} | {r['dmv']:.1f} | {r['snr_exp']:.1f} |" for r in missed]
(D / "alerts.md").write_text("\n".join(o) + "\n")
print("\n".join(o))
