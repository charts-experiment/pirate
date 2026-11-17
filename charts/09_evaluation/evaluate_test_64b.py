#!/usr/bin/env python
"""Evaluation of the 64-beam test: recall + false positives.

Inputs (in the run folder):
  assignment.csv   ground truth: beam and start sample of each file within its stream
  sifter.log       events of the search (format of pirate's toy_sifter)
  emitter.json, server.log, gpu.csv, vmstat.log   performance
(Runs made with the earlier Spanish-named chain, with asignacion.csv and emisor.json, are also read.)

Criteria (the same as in the offline benchmark):
  - search from DM 20 (events with DM < 20 are discarded, like --dm-min 20)
  - A/B/P hit: same beam, |DM - true DM| <= max(10 %, 1) and |t - true t| <= max(50 ms, optimal boxcar),
    with t = arrival at 501.6 MHz (the pirate event carries the arrival at the bottom edge of the band;
    it is moved to 501.6 MHz with the delay of its own DM)
  - non-FRB FP: % of negatives with any event (any DM) within ±100 ms of t_top
  - noise FP: events outside [t_top - sweep - 2 s, t_top + sweep + 2 s] of all files of the beam
    (and outside the initial padding), per hour of signal-free time, summed over the 64 beams
  - Recall over files with DM <= 1000 (requirement of the test); also over the whole search range
Reported at S/N >= 5, 6, 7, 8 and at equal noise FP rate (100, 30, 10 and 1 per hour).

Usage: python evaluate_test_64b.py RUN_FOLDER
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

if len(sys.argv) < 2:
    sys.exit(__doc__)
D = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
# dedispersion config: the one the launcher saved in the run, or the one in the repository
DD = D / "config_dedispersion.yml" if (D / "config_dedispersion.yml").exists() else HERE.parent / "06_configs" / "charts_64b_dm1000.yml"
SEQ = 200                     # seq per sample (1.000 ms clock of the emitter)
DM_MIN = 20.0
TOL_DM, TOL_T, TOL_NEG = 0.10, 0.05, 0.10


def first_existing(*names):
    for n in names:
        if (D / n).exists():
            return D / n
    sys.exit(f"ERROR: none of {names} in {D}")


def delay_edge_to_top():
    """Seconds per unit DM between the bottom edge of the band (where pirate timestamps the event) and 501.6 MHz."""
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    s_per_dm_band = 1e-3 / cfg.dm_per_unit_delay()
    return s_per_dm_band * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2), cfg


K_TOP, CFG = delay_edge_to_top()
DM_MAX = max(t.dm_max for t in __import__("pirate_frb").DedispersionPlan(CFG).trees)

# ---- events ----
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
ev = []
for line in (D / "sifter.log").read_text(errors="replace").splitlines():
    m = ev_re.search(line)
    if m:
        ev.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
ev = np.array(ev, dtype=float).reshape(-1, 5)
ev = ev[ev[:, 1] >= DM_MIN]
ev_beam, ev_dm, ev_snr = ev[:, 0].astype(int), ev[:, 1], ev[:, 4]
ev_t = ev[:, 2] / SEQ * 1e-3 - ev_dm * K_TOP            # arrival at 501.6 MHz (s, since the start of the stream)

# ---- ground truth ----
rows = list(csv.DictReader(open(first_existing("assignment.csv", "asignacion.csv"))))
em = json.loads(first_existing("emitter.json", "emisor.json").read_text())
nbeams = em["nbeams"]
total_s = em.get("total_samples", em.get("total_muestras")) / 1000
prepad_s = em["prepad"] / 1000
for r in rows:
    r["b"] = int(r["beam"])
    r["t0"] = int(r.get("start_sample") or r["inicio_muestra"]) / 1000
    r["t_abs"] = r["t0"] + float(r["t_top_s"])
    r["dmv"] = float(r["dm_equivalente_pos"] if r["set"] == "NEG" else r["dm"])
    r["sweep"] = float(r["barrido_s"])
    r["box"] = float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000
    r["snr_exp"] = float(r.get("snr_esperado") or r.get("snr_visible_esperado") or "nan")

# best S/N of an event that hits each file (or -inf)
best = np.full(len(rows), -np.inf)
for i, r in enumerate(rows):
    sel = ev_beam == r["b"]
    if r["set"] == "NEG":
        ok = sel & (np.abs(ev_t - r["t_abs"]) <= TOL_NEG)
    else:
        ok = sel & (np.abs(ev_dm - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0)) & \
             (np.abs(ev_t - r["t_abs"]) <= max(TOL_T, r["box"]))
    if ok.any():
        best[i] = ev_snr[ok].max()

# noise: events outside all signal windows; signal-free time per beam
noise_snr, free = [], 0.0
for b in range(nbeams):
    win = sorted([(r["t_abs"] - r["sweep"] - 2, r["t_abs"] + r["sweep"] + 2) for r in rows if r["b"] == b])
    win = [(0.0, prepad_s)] + win
    sel = ev_beam == b
    t = ev_t[sel]
    outside = np.ones(t.size, bool)
    for lo, hi in win:
        outside &= ~((t >= lo) & (t <= hi))
    noise_snr.append(ev_snr[sel][outside])
    union, (clo, chi) = 0.0, win[0]
    for lo, hi in win[1:]:
        if lo > chi:
            union += chi - clo
            clo, chi = lo, hi
        else:
            chi = max(chi, hi)
    union += chi - clo
    free += total_s - min(union, total_s)
noise_snr = np.concatenate(noise_snr)
hours = free / 3600

sets = np.array([r["set"] for r in rows])
dmv = np.array([r["dmv"] for r in rows])
in_range = dmv <= DM_MAX / (1 - TOL_DM)
d1000 = dmv <= 1000


def row(u):
    out = {"S/N threshold": u, "noise FP/h": round(float((noise_snr >= u).sum() / hours), 1)}
    for s, name in (("A", "A"), ("B", "B"), ("P", "partial")):
        m = (sets == s) & d1000
        out[f"Recall {name} DM≤1000 (%)"] = round(100 * float((best[m] >= u).mean()), 1)
    out["non-FRB FP (% NEG DM≤1000)"] = round(100 * float((best[(sets == "NEG") & d1000] >= u).mean()), 1)
    for s in ("A", "B"):
        m = (sets == s) & in_range
        out[f"Recall {s} DM≤{DM_MAX:.0f} (%)"] = round(100 * float((best[m] >= u).mean()), 1)
    return out


thresholds = np.round(np.arange(5.0, 15.0, 0.02), 2)
fp_curve = np.array([(noise_snr >= u).sum() / hours for u in thresholds])
tab_fixed = [row(u) for u in (5.0, 6.0, 7.0, 8.0)]
tab_fp = []
for target in (100, 30, 10, 1):
    k = np.flatnonzero(fp_curve <= target)
    if len(k):
        tab_fp.append({"target noise FP/h": target, **row(float(thresholds[k[0]]))})


def md_table(table):
    cols = list(table[0])
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for f in table:
        s += "| " + " | ".join(str(f[c]) for c in cols) + " |\n"
    return s


# ---- performance ----
srv = [l for l in (D / "server.log").read_text(errors="replace").splitlines() if "beams=" in l]
beq = np.array([float(l.split("beams=")[1].split("/")[0]) for l in srv]) if srv else np.array([np.nan])
gbps = np.array([float(l.split("Gbps=")[1].split(",")[0]) for l in srv]) if srv else np.array([np.nan])
gl = [l.split(",") for l in (D / "gpu.csv").read_text().splitlines()[1:] if l.count(",") >= 3]
gu = np.array([float(x[1].strip().split()[0]) for x in gl]) if gl else np.array([np.nan])
gm = np.array([float(x[2].strip().split()[0]) for x in gl]) if gl else np.array([np.nan])
vm = [l.split() for l in (D / "vmstat.log").read_text().splitlines()[2:] if l.split() and l.split()[0].isdigit()]
idle = np.array([float(x[14]) for x in vm]) if vm else np.array([np.nan])
free_ram = np.array([float(x[3]) / 1048576 for x in vm]) if vm else np.array([np.nan])

rate = em.get("measured_rate", em.get("ritmo_medido", float("nan")))
duration = em.get("duration_s", em.get("duracion_s", float("nan")))
n = {s: int(((sets == s) & d1000).sum()) for s in ("A", "B", "NEG", "P")}
txt = f"""# 64-beam test with pirate — {D.name}

- Files: {len(rows)} (group 3, 21 504 channels) in {nbeams} beams; {total_s:.0f} s of stream per beam.
- Search: DM {DM_MIN:.0f}–{DM_MAX:.0f}, boxcar ≤ 32 ms, rank 15 (one tree), 4 bits.
- With DM ≤ 1000: A {n['A']}, B {n['B']}, NEG {n['NEG']}, partial {n['P']}.
- Signal-free time for the noise FP: {hours:.2f} h (64 beams summed). Search events (DM ≥ 20): {len(ev)}.

## At fixed threshold

{md_table(tab_fixed)}
## At equal noise FP rate

{md_table(tab_fp) if tab_fp else '(not enough data)'}
## Performance

| | |
|---|---|
| Emitter rate | {rate:.2f}× real time ({duration:.0f} s for {total_s:.0f} s of data) |
| Server capacity (line beams=X/{nbeams}) | median {np.nanmedian(beq):.1f} equivalent beams ({np.nanmedian(beq) / nbeams:.2f}× real time) |
| Input | median {np.nanmedian(gbps):.1f} Gbps |
| GPU | mean use {np.nanmean(gu):.0f} %, max VRAM {np.nanmax(gm):.0f} MiB |
| Idle CPU | mean {np.nanmean(idle):.0f} % |
| Minimum free RAM | {np.nanmin(free_ram):.1f} GB |
"""
(D / "results.md").write_text(txt)
np.save(D / "best_snr_per_file.npy", best)
print(txt)
