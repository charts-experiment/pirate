#!/usr/bin/env python
"""T3b · Statistics of pirate's DM-time map: where signal and noise cross.

Uses the full run of grouper v1 (T2), without running anything new:
  events_v0.log    maximum per chunk and beam (almost every beam-chunk has a maximum ≥ 5, so it is the
                   full distribution of the maximum per chunk above 5)
  sifter.log       v1 events (up to 4 per beam and chunk, S/N > 5)
  assignment.csv   ground truth
(Runs of the earlier Spanish-named chain, with eventos_v0.log, asignacion.csv and emisor.json, are also read.)

Analyses:
  1. Noise: distribution of the maximum per beam-chunk in beam-chunks without signal (no piece of the diagonal of
     any file). Fit of N_eff (equivalent independent gaussian trials): P(max ≥ T) = 1 − (1 − Q(T))^N_eff.
  2. Noise versus DM and width: are the noise peaks spread evenly in DM or concentrated?
  3. Signal: S/N measured by pirate against the expected (injected) S/N, and recall versus expected S/N.
  4. Crossing: at which expected S/N an FRB falls in the noise region.

Usage (pirate environment): python map_statistics.py [RUN_FOLDER]   (or export CHARTS_RUN=...)
Output: RUN/analysis/ (or $CHARTS_ANALYSIS_OUT): metrics.json, tables.md and 4 figures
"""
import csv
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 2 and "CHARTS_RUN" not in os.environ:
    sys.exit(f"Usage: python {sys.argv[0]} RUN_FOLDER   (or export CHARTS_RUN=...)")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["CHARTS_RUN"])   # full run with events_v0.log
OUTPUT = Path(os.environ.get("CHARTS_ANALYSIS_OUT", RUN / "analysis"))   # tables and figures
OUTPUT.mkdir(parents=True, exist_ok=True)
DD = Path(__file__).resolve().parents[2] / "06_configs" / "charts_64b_dm1000.yml"
SEQ, DM_MIN, TOL_DM, TOL_T = 200, 20.0, 0.10, 0.05
T_CHUNK = 2.048
OP_THRESHOLD = 6.50                                # 10 noise FP/h in T2
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
V = plt.cm.viridis


def first_existing(*names):
    for n in names:
        if (RUN / n).exists():
            return RUN / n
    sys.exit(f"ERROR: none of {names} in {RUN}")


def delay_edge_to_top():
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    s_per_dm_band = 1e-3 / cfg.dm_per_unit_delay()
    dm_max = max(t.dm_max for t in pf.DedispersionPlan(cfg).trees)
    return s_per_dm_band * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2), s_per_dm_band, dm_max


K_TOP, K_BAND, DM_MAX = delay_edge_to_top()
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
ch_re = re.compile(r"fpga=\[(-?\d+):(-?\d+)\]")


def read_log(p, with_chunks=False):
    ev, chunks, ic = [], [], -1
    for line in p.read_text(errors="replace").splitlines():
        m = ch_re.search(line)
        if m:
            ic += 1
            chunks.append(int(m.group(1)) / SEQ * 1e-3)
            continue
        m = ev_re.search(line)
        if m:
            ev.append((int(m.group(1)), float(m.group(2)), int(m.group(3)) / SEQ * 1e-3, float(m.group(4)),
                       float(m.group(5)), ic))
    ev = np.array(ev, dtype=float).reshape(-1, 6)
    return (ev, np.array(chunks)) if with_chunks else ev


# ---- ground truth ----
rows = list(csv.DictReader(open(first_existing("assignment.csv", "asignacion.csv"))))
emitter = json.loads(first_existing("emitter.json", "emisor.json").read_text())
NB, PREPAD = emitter["nbeams"], emitter["prepad"] / 1000
for r in rows:
    start = int(r.get("start_sample") or r["inicio_muestra"]) / 1000
    r["b"] = int(r["beam"])
    r["t_abs"] = start + float(r["t_top_s"])
    r["dmv"] = float(r["dm_equivalente_pos"] if r["set"] == "NEG" else r["dm"])
    r["box"] = float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000
    r["snr_exp"] = float(r.get("snr_visible_esperado") or r.get("snr_esperado") or "nan")
    r["t_start"] = start
    r["t_end"] = start + float(r["largo_s"])

# ---- 1. noise: maximum per beam-chunk without signal ----
ev0, chunk_t0 = read_log(first_existing("events_v0.log", "eventos_v0.log"), with_chunks=True)
nchunks = len(chunk_t0)
# A beam-chunk is "without signal" if its interval (time at the bottom edge of the band, which is how pirate
# timestamps) does not touch [file start − 2 s, file end + sweep + 2 s] of any file of the beam:
# so no piece of a diagonal (at any trial DM) and no initial padding gets in.
clean = np.ones((NB, nchunks), bool)
c0 = chunk_t0
c1 = chunk_t0 + T_CHUNK
clean[:, c1 <= PREPAD + DM_MAX * K_BAND] = False
for r in rows:
    lo, hi = r["t_start"] - 2, r["t_end"] + DM_MAX * K_BAND + 2
    clean[r["b"], (c1 > lo) & (c0 < hi)] = False
clean[:, -2:] = False                               # last chunks (shutdown)
maxchunk = np.full((NB, nchunks), np.nan)
info = {}
for b, dm, t, w, s, ic in ev0:
    if dm >= DM_MIN and 0 <= ic < nchunks:
        maxchunk[int(b), int(ic)] = s
        info[(int(b), int(ic))] = (dm, w)
sel = clean & ~np.isnan(maxchunk)
n_clean = int(clean.sum())
m_noise = maxchunk[sel]
missing = n_clean - m_noise.size                     # clean beam-chunks without a maximum ≥ 5 (or with DM < 20)
print(f"beam-chunks without signal: {n_clean} ({n_clean / clean.size:.1%}); with a recorded maximum: {m_noise.size}")

T = np.round(np.arange(5.0, 8.01, 0.05), 2)
surv = np.array([(m_noise >= u).sum() / n_clean for u in T])
chunks_per_h = 3600 / T_CHUNK
fp_h = surv * chunks_per_h * NB


def p_max(u, logn):
    q = stats.norm.sf(u)
    return -np.expm1(10 ** logn * np.log1p(-q))


ok = (surv > 0) & (T >= 5.5)
res = optimize.minimize_scalar(lambda ln: np.sum((np.log(p_max(T[ok], ln)) - np.log(surv[ok])) ** 2),
                               bounds=(4, 12), method="bounded")
LOGN = res.x
print(f"N_eff ≈ 10^{LOGN:.2f} gaussian trials per beam-chunk")

# ---- 2. noise versus DM and width ----
dm_r = np.array([info[(b, c)][0] for b, c in zip(*np.nonzero(sel))])
w_r = np.array([info[(b, c)][1] for b, c in zip(*np.nonzero(sel))])
bins_dm = np.linspace(DM_MIN, DM_MAX, 12)
idx = np.digitize(dm_r, bins_dm) - 1
per_dm = []
for k in range(len(bins_dm) - 1):
    mk = idx == k
    per_dm.append(dict(dm_lo=bins_dm[k], dm_hi=bins_dm[k + 1], n=int(mk.sum()),
                       frac=float(mk.mean()), frac_ge_6=float((m_noise[mk] >= 6).mean()) if mk.any() else np.nan,
                       median=float(np.median(m_noise[mk])) if mk.any() else np.nan))
widths = np.unique(w_r)
per_width = [dict(width_ms=float(a), n=int((w_r == a).sum()), frac=float((w_r == a).mean()),
                  median=float(np.median(m_noise[w_r == a]))) for a in widths]
per_beam = np.array([np.nanmedian(np.where(sel[b], maxchunk[b], np.nan)) for b in range(NB)])

# ---- 3. signal: best v1 S/N that hits each file ----
ev1 = read_log(RUN / "sifter.log")
ev1 = ev1[ev1[:, 1] >= DM_MIN]
e_b, e_dm, e_s = ev1[:, 0].astype(int), ev1[:, 1], ev1[:, 4]
e_t = ev1[:, 2] - e_dm * K_TOP
best = np.full(len(rows), np.nan)
width_det = np.full(len(rows), np.nan)
for i, r in enumerate(rows):
    if r["set"] == "NEG":
        continue
    ok_ = (e_b == r["b"]) & (np.abs(e_dm - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0)) & \
          (np.abs(e_t - r["t_abs"]) <= max(TOL_T, r["box"]))
    best[i] = e_s[ok_].max() if ok_.any() else 0.0
    if ok_.any():
        width_det[i] = ev1[ok_, 3][np.argmax(e_s[ok_])]
sets = np.array([r["set"] for r in rows])
dmv = np.array([r["dmv"] for r in rows])
exp = np.array([r["snr_exp"] for r in rows])
use = (sets != "NEG") & (dmv <= 1000) & np.isfinite(exp)

bins_s = np.array([0, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 1e9])
recall = []
for k in range(len(bins_s) - 1):
    m = use & (exp >= bins_s[k]) & (exp < bins_s[k + 1])
    det = best[m]
    recall.append(dict(snr_exp_lo=bins_s[k], snr_exp_hi=bins_s[k + 1], n=int(m.sum()),
                       recall_5=float((det >= 5).mean()) if m.any() else np.nan,
                       recall_op=float((det >= OP_THRESHOLD).mean()) if m.any() else np.nan,
                       median_measured=float(np.median(det[det > 0])) if (det > 0).any() else np.nan))
det = use & (best > 0) & (exp >= 6) & (exp <= 30) & (sets != "P")
ratio = best[det] / exp[det]


# thresholds for a given rate, with the fitted gaussian model (the measured one lacks statistics)
def threshold(fp_target, nbeams):
    return float(optimize.brentq(lambda u: p_max(u, LOGN) * chunks_per_h * nbeams - fp_target, 5, 10))


points = []
for name, fp_target, nb in (("10 FP per beam·hour (T1–T3 criterion)", 10, 1), ("10 FP/h of the system (64 beams)", 10, NB),
                            ("1 FP/h of the system (64 beams)", 1, NB)):
    u = threshold(fp_target, nb)
    row_p = {"criterion": name, "threshold": round(u, 2)}
    for s_, n_ in (("A", "A"), ("B", "B"), ("P", "cut")):
        m = (sets == s_) & (dmv <= 1000)
        row_p[f"recall {n_}"] = round(100 * float((best[m] >= u).mean()), 1)
    points.append(row_p)
w_frb = width_det[use & (best >= OP_THRESHOLD)]

# ---- tables ----
summary = {
    "run": str(RUN),
    "beam_chunks_without_signal": n_clean,
    "beam_chunks_without_recorded_maximum": int(missing),
    "log10_N_eff": round(float(LOGN), 2),
    "noise_maximum_percentiles": {p: round(float(np.percentile(m_noise, p)), 2) for p in (50, 90, 99, 99.9)},
    "gauss_threshold_10FPh": round(float(optimize.brentq(lambda u: p_max(u, LOGN) * chunks_per_h * NB - 10, 5, 9)), 2),
    "measured_threshold_10FPh": float(T[np.flatnonzero(fp_h <= 10)[0]]) if (fp_h <= 10).any() else None,
    "measured_expected_ratio_median": round(float(np.median(ratio)), 3),
    "measured_expected_ratio_p10_p90": [round(float(np.percentile(ratio, p)), 3) for p in (10, 90)],
    "operating_points": points,
    "per_dm": per_dm, "per_width": per_width, "recall_per_expected_snr": recall,
    "per_beam_median_min_max": [round(float(per_beam.min()), 2), round(float(per_beam.max()), 2)],
}
(OUTPUT / "metrics.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


def md(rows_, cols, fmt):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return s + "".join("| " + " | ".join(f(r) for f in fmt) + " |\n" for r in rows_)


txt = "# T3b · tables\n\n## Threshold and recall versus the FP criterion (fitted gaussian model)\n\n"
txt += md(points, list(points[0]), [lambda r, c=c: str(r[c]) for c in points[0]])
txt += "\n## Noise maximum per beam-chunk\n\n"
txt += md([dict(u=u, s=s, f=f) for u, s, f in zip(T, surv, fp_h) if abs(u * 4 - round(u * 4)) < 1e-6],
          ["S/N ≥", "P per beam-chunk", "FP/h of the system (64 beams)", "gauss N_eff"],
          [lambda r: f"{r['u']:.2f}", lambda r: f"{r['s']:.2e}", lambda r: f"{r['f']:.1f}",
           lambda r: f"{p_max(r['u'], LOGN):.2e}"])
txt += "\n## Noise versus DM (maxima of beam-chunks without signal)\n\n"
txt += md(per_dm, ["DM", "n", "fraction", "median S/N", "P(S/N ≥ 6)"],
          [lambda r: f"{r['dm_lo']:.0f}–{r['dm_hi']:.0f}", lambda r: str(r["n"]), lambda r: f"{r['frac']:.1%}",
           lambda r: f"{r['median']:.2f}", lambda r: f"{r['frac_ge_6']:.3f}"])
txt += "\n## Noise and detected FRBs versus width\n\n"
txt += md(per_width, ["width (ms)", "noise n", "noise fraction", "noise median S/N", f"detected FRBs (S/N ≥ {OP_THRESHOLD}) fraction"],
          [lambda r: f"{r['width_ms']:g}", lambda r: str(r["n"]), lambda r: f"{r['frac']:.1%}",
           lambda r: f"{r['median']:.2f}", lambda r: f"{(w_frb == r['width_ms']).mean():.1%}"])
txt += "\n## Recall versus expected S/N (A, B and cut, DM ≤ 1000)\n\n"
txt += md(recall, ["expected S/N", "n", "recall S/N ≥ 5", f"recall S/N ≥ {OP_THRESHOLD}", "median measured S/N"],
          [lambda r: f"{r['snr_exp_lo']:g}–{r['snr_exp_hi']:g}" if r["snr_exp_hi"] < 1e8 else f"≥ {r['snr_exp_lo']:g}",
           lambda r: str(r["n"]), lambda r: f"{r['recall_5']:.1%}", lambda r: f"{r['recall_op']:.1%}",
           lambda r: f"{r['median_measured']:.2f}"])
(OUTPUT / "tables.md").write_text(txt)
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=1, ensure_ascii=False))
print(txt)

# ---- figures ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axs[0]
ax.hist(m_noise, bins=np.arange(5, 8.5, 0.05), color=V(0.3), alpha=0.85, label="maximum per beam-chunk (noise)")
ax.axvline(OP_THRESHOLD, color="black", ls="--", lw=1)
ax.text(OP_THRESHOLD + 0.05, ax.get_ylim()[1] * 0.9, "6.5: 10 per beam·hour", fontsize=9)
u_sys = threshold(10, NB)
ax.axvline(u_sys, color="black", ls=":", lw=1)
ax.text(u_sys + 0.05, ax.get_ylim()[1] * 0.75, f"{u_sys:.2f}: 10 per hour of the system", fontsize=9)
ax.set_xlabel("S/N")
ax.set_ylabel("beam-chunks")
ax.set_title(f"Noise: maximum per beam-chunk ({n_clean} beam-chunks without signal)")
ax = axs[1]
ax.semilogy(T, fp_h, "o", ms=4, color=V(0.5), label="measured")
tt = np.linspace(5, 8, 200)
for ln, ls in ((LOGN, "-"), (LOGN - 1, ":"), (LOGN + 1, ":")):
    ax.semilogy(tt, p_max(tt, ln) * chunks_per_h * NB, ls, color=V(0.0),
                label=f"gaussian, N_eff = 10^{ln:.1f}" if ln == LOGN else (f"10^{ln:.1f}"))
ax.axhline(10, color="gray", lw=0.8)
ax.axhline(10 * NB, color="gray", lw=0.8, ls="--")
ax.text(5.05, 10 * NB * 1.3, "10 per beam·hour", fontsize=8, color="gray")
ax.text(5.05, 13, "10 per hour of the system", fontsize=8, color="gray")
ax.set_ylim(0.3, 3e5)
ax.set_xlabel("S/N threshold")
ax.set_ylabel("noise FP per hour, 64-beam system")
ax.set_title("False alarm rate versus threshold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT / "1_noise_maximum_per_chunk.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
centres = [(r["dm_lo"] + r["dm_hi"]) / 2 for r in per_dm]
axs[0].bar(centres, [r["frac"] for r in per_dm], width=np.diff(bins_dm)[0] * 0.9, color=V(0.3))
axs[0].axhline(1 / (len(bins_dm) - 1), color="black", ls="--", lw=1, label="even split")
axs[0].set_xlabel("DM of the noise maximum")
axs[0].set_ylabel("fraction of beam-chunks")
axs[0].legend(fontsize=8)
axs[0].set_title("Where does the noise maximum fall in DM?")
axs[1].plot(centres, [r["frac_ge_6"] for r in per_dm], "o-", color=V(0.5))
axs[1].set_xlabel("DM")
axs[1].set_ylabel("P(maximum ≥ 6 | DM)")
axs[1].set_title("Noise tail versus DM")
xa = np.arange(len(per_width))
axs[2].bar(xa - 0.2, [r["frac"] for r in per_width], width=0.4, color=V(0.3), label="noise")
axs[2].bar(xa + 0.2, [(w_frb == r["width_ms"]).mean() for r in per_width], width=0.4, color=V(0.95),
           edgecolor="0.4", lw=0.3, label="detected FRBs")
axs[2].set_xticks(xa, [f"{r['width_ms']:g}" for r in per_width])
axs[2].set_xlabel("width (ms)")
axs[2].set_ylabel("fraction")
axs[2].legend(fontsize=8)
axs[2].set_title("Noise and FRBs versus width")
fig.tight_layout()
fig.savefig(OUTPUT / "2_noise_dm_width.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]
colours = {"A": V(0.0), "B": V(0.5), "P": V(0.95)}
for s in ("B", "A", "P"):
    m = use & (sets == s)
    y = np.where(best[m] > 0, best[m], 4.2)
    ax.scatter(exp[m], y, s=9, alpha=0.6, color=colours[s], edgecolor="none" if s != "P" else "0.4",
               linewidths=0.3, label={"A": "A (S/N 8)", "B": "B", "P": "cut (visible S/N)"}[s])
xx = np.array([1, 300])
ax.plot(xx, xx, color="black", lw=0.8, label="measured = expected")
ax.axhline(OP_THRESHOLD, color="gray", ls="--", lw=1)
ax.axhspan(4, 4.4, color="0.9")
ax.text(1.1, 4.25, "not detected", fontsize=8, va="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 300)
ax.set_ylim(4, 300)
ax.set_xlabel("expected S/N (injected)")
ax.set_ylabel("best S/N measured by pirate (v1)")
ax.set_title(f"Measured / expected: median {np.median(ratio):.2f}")
ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
xc = [np.sqrt(max(r["snr_exp_lo"], 2) * min(r["snr_exp_hi"], 100)) for r in recall]
ax.semilogx(xc, [100 * r["recall_5"] for r in recall], "o-", color=V(0.3), label="S/N ≥ 5")
ax.semilogx(xc, [100 * r["recall_op"] for r in recall], "o-", color=V(0.6), label=f"S/N ≥ {OP_THRESHOLD} (10 FP per beam·hour)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([3, 5, 7, 10, 15, 20, 30, 50, 70])
ax.set_xlabel("expected S/N (per bin)")
ax.set_ylabel("recall (%)")
ax.set_ylim(0, 102)
ax.set_title("Recall versus expected S/N (A + B + cut, DM ≤ 1000)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT / "3_signal_measured_vs_expected.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.hist(m_noise, bins=np.arange(4, 16, 0.1), density=True, color=V(0.3), alpha=0.6, label="noise: maximum per beam-chunk")
d8 = best[use & (sets == "A") & (best > 0)]
ax.hist(d8, bins=np.arange(4, 16, 0.1), density=True, color=V(0.95), alpha=0.7, edgecolor="0.4", lw=0.3,
        label="set A FRBs (S/N 8 injected), measured")
ax.axvline(OP_THRESHOLD, color="black", ls="--", lw=1)
ax.set_xlabel("measured S/N")
ax.set_ylabel("density")
ax.set_title("Crossing between noise and an S/N 8 FRB")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT / "4_noise_signal_crossing.png", dpi=130)
plt.close(fig)
