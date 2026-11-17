#!/usr/bin/env python3
"""Count of ALL candidates (grouper v1 without the --k/--k-total caps) with and without the site RFI.

For each run (corta_todos_rfi and corta_todos_sin_rfi, the names of the existing runs) it reads all_events.log
(sifter.log format; eventos_todos.log in older runs) and sifter.log (what reached the sifter, with the cap of 4 per
beam and 256 per chunk), and reports:
  - how many events there are: total, per chunk, per beam and chunk, per hour;
  - what the events outside the signal windows look like: S/N, DM, width;
  - how many appear in many beams at once (same t_top ± 50 ms in ≥ 8 / ≥ 32 beams);
  - Recall and FP with all the events: <run>/all/ is built with sifter.log → all_events.log and
    evaluate_test_64b.py is run (the usual criteria).
Usage (pirate environment): export CHARTS_RFI_RESULTS=FOLDER; python analyze_all.py
FOLDER contains results/<run> (or resultados/<run>). Output: FOLDER/all_counts.md, FOLDER/all_counts.png
"""
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import R, RUNS_DIR, K_TOP, existing, all_events_log, start_s, evaluate as _evaluate

EV = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
CH = re.compile(r"fpga=\[(\d+):(\d+)\]")
SEQ, T = 200, 2048


def read_events(log):
    txt = Path(log).read_text(errors="replace")
    e = np.array([[float(x) for x in m.groups()] for m in EV.finditer(txt)]) if EV.search(txt) else np.zeros((0, 5))
    nch = len(set(CH.findall(txt)))
    return e, nch


def signal_windows(d):
    """Intervals [t0, t1] (s of the stream) per beam that contain signal: file ± sweep ± 2 s, like the evaluator."""
    v = {}
    for r in csv.DictReader(open(existing(d, "assignment.csv", "asignacion.csv"))):
        t0 = start_s(r) + float(r["t_top_s"])
        sw = float(r.get("barrido_s") or 0)
        v.setdefault(int(r["beam"]), []).append((t0 - sw - 2, t0 + sw + 2))
    return v


def outside_signal(e, v):
    t = e[:, 2] / SEQ * 1e-3 - e[:, 1] * K_TOP
    m = np.ones(len(e), bool)
    for b, iv in v.items():
        sel = e[:, 0] == b
        for a, z in iv:
            m[sel & (t >= a) & (t <= z)] = False
    return m, t


def multibeam(e, t, tol=0.05):
    """For each event, in how many distinct beams there is an event with the same t_top ± tol."""
    o = np.argsort(t); ts, bs = t[o], e[o, 0].astype(int)
    n = np.zeros(len(e), int)
    j0 = 0
    for i in range(len(ts)):
        while ts[j0] < ts[i] - tol:
            j0 += 1
        j1 = np.searchsorted(ts, ts[i] + tol, "right")
        n[o[i]] = len(set(bs[j0:j1]))
    return n


def evaluate(d, log, label):
    return _evaluate(d, log, label)[0]


def table(md):
    """Rows of the two Recall/FP tables of results.md."""
    return [l for l in md.splitlines() if re.match(r"^\| [0-9]", l)]


txt = ["# Count of all candidates with and without the site RFI (grouper v1 without caps)\n",
       f"Event threshold: S/N > 6 (`-t 6`). `--max-cand 512` candidates decoded per beam and chunk. "
       f"Time of each event moved to 501.6 MHz with K = {K_TOP * 1e3:.2f} ms per unit of DM.\n"]
res = {}
for v in ("sin_rfi", "rfi"):
    d = RUNS_DIR / f"corta_todos_{v}"
    e, nch = read_events(all_events_log(d))
    es, _ = read_events(d / "sifter.log")
    vs = signal_windows(d)
    outside, t = outside_signal(e, vs)
    nb = multibeam(e, t)
    hours = nch * T / 1000 / 3600
    ich = (e[:, 2] // (T * SEQ)).astype(int)
    per_chunk = np.bincount(ich)
    pbc = np.unique(ich * 1000 + e[:, 0].astype(int), return_counts=True)[1]
    res[v] = dict(e=e, outside=outside, nb=nb)
    txt += [f"\n## {'With site RFI' if v == 'rfi' else 'No RFI (control, with normalization)'} (`{d.name}`)\n",
            "| | |", "|---|---|",
            f"| Chunks | {nch} ({hours * 3600:.0f} s of stream per beam; {hours * 64:.2f} beam·hour) |",
            f"| Total events (all) | {len(e)} |",
            f"| Events that reached the sifter (cap 4/beam, 256/chunk) | {len(es)} |",
            f"| Events per chunk (all): median / max | {np.median(per_chunk):.0f} / {per_chunk.max()} |",
            f"| Events per beam and chunk: median / p90 / max | {np.median(pbc):.0f} / {np.percentile(pbc, 90):.0f} / {pbc.max()} |",
            f"| Fraction of beam-chunks with > 4 events (those the cap cut) | {(pbc > 4).sum() / (nch * 64) * 100:.1f} % |",
            f"| Events outside the signal windows | {outside.sum()} ({outside.sum() / (hours * 64):.0f} per beam·hour) |",
            f"| … of them with DM < 100 | {(outside & (e[:, 1] < 100)).mean() / max(outside.mean(), 1e-9) * 100:.0f} % |",
            f"| … in ≥ 8 beams at once (±50 ms) | {(outside & (nb >= 8)).sum() / max(outside.sum(), 1) * 100:.0f} % |",
            f"| … in ≥ 32 beams at once | {(outside & (nb >= 32)).sum() / max(outside.sum(), 1) * 100:.0f} % |",
            f"| S/N outside signal: p50 / p90 / p99 / max | " + " / ".join(f"{x:.1f}" for x in np.percentile(e[outside, 4], [50, 90, 99, 100])) + " |" if outside.any() else "| S/N outside signal | — |",
            f"| Width outside signal (ms): p50 / p90 | " + " / ".join(f"{x:.0f}" for x in np.percentile(e[outside, 3], [50, 90])) + " |" if outside.any() else "| Width | — |"]
    for label, log in (("capped", d / "sifter.log"), ("all", all_events_log(d))):
        md = evaluate(d, log, label)
        txt += [f"\n**Recall and FP with the events {'that reached the sifter (capped)' if label == 'capped' else 'of all (no cap)'}** "
                f"(DM ≤ 1000; noise FP per beam·hour):\n",
                "| S/N threshold | noise FP/h | Recall A | Recall B | Recall cut | non-FRB FP % | Recall A DM≤1105 | Recall B DM≤1105 |",
                "|---|---|---|---|---|---|---|---|"] + table(md)[:4]
        fp = table(md)[4:]
        if fp:
            txt += ["\nAt equal noise FP (target / threshold / FP / Recall A / B / cut / non-FRB FP / A≤1105 / B≤1105):\n",
                    "| FP target | threshold | FP/h | A | B | cut | non-FRB % | A≤1105 | B≤1105 |", "|---|---|---|---|---|---|---|---|---|"] + fp
(R / "all_counts.md").write_text("\n".join(txt) + "\n")
print("\n".join(txt))

# figure
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
col = {"sin_rfi": plt.cm.viridis(0.25), "rfi": plt.cm.viridis(0.8)}
name = {"sin_rfi": "no RFI", "rfi": "with site RFI"}
for v, r in res.items():
    e, f = r["e"], r["outside"]
    if not f.any():
        continue
    ax[0, 0].hist(e[f, 4], bins=np.geomspace(6, 300, 50), histtype="step", lw=2, color=col[v], label=f"{name[v]} ({f.sum()})")
    ax[0, 1].hist(e[f, 1], bins=np.linspace(0, 1105, 56), histtype="step", lw=2, color=col[v], label=name[v])
    ax[1, 0].hist(e[f, 3], bins=np.geomspace(1, 64, 20), histtype="step", lw=2, color=col[v], label=name[v])
    ax[1, 1].hist(r["nb"][f], bins=np.arange(1, 66), histtype="step", lw=2, color=col[v], label=name[v])
for a, (xl, lg) in zip(ax.ravel(), (("S/N", True), ("DM (pc cm⁻³)", False), ("Width (ms)", True), ("Beams with the same event (±50 ms)", False))):
    a.set(xlabel=xl, ylabel="events outside the signal windows", yscale="log")
    if lg:
        a.set_xscale("log")
    a.legend(); a.grid(alpha=0.3)
fig.suptitle("All grouper v1 candidates (no cap) outside the signal windows: with and without the site RFI", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(R / "all_counts.png", dpi=110)
