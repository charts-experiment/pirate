#!/usr/bin/env python3
"""
Pipeline comparison set (design agreed in the CHARTS team, 2026-09-23). Group 2: 672 channels.

The earlier set (outputs/) inherits the flux of the CHIME catalog (the amplitude is not
controlled), searches DM only up to 1000 (14% of the positives fall outside) and
does not include the dispersion smearing within each channel. This set
controls the amplitude explicitly:

  A_snr_fijo/        N FRBs, all with the SAME expected S/N (optimal
                     boxcar, dedispersed at the true DM). DM, width,
                     scattering and spectrum differ in each one.
  B_amplitud_fija/   The SAME N parameter draws as A, but all
                     with the same amplitude S0 (Jy): intrinsic peak flux
                     (before scattering and smearing) in the brightest
                     channel of the CHARTS band. The S/N follows from the physics.
                     S0 is calibrated so that the median S/N is
                     --snr-b-median.
  negativos/         N non-FRBs, paired with A: same parameters and same
                     scale as A_i, but with an invalid dm_index (|idx| in
                     [1.5,1.8) or (2.2,2.5]) and the DM rescaled so that the
                     sweep crosses the band in the same time as the positive
                     (it does not leave the file).
  parcial/           N_partial FRBs cut by the edge of the file: the
                     sweep starts before the beginning ("inicio": only the
                     low part of the band remains) or ends after the end
                     ("fin": only the high part remains). The visible energy
                     fraction is drawn (uniform 5-95%) and the cut is placed
                     to obtain it. The full burst is normalized to
                     --snr-partial; the manifest stores the visible band,
                     the energy fraction inside and the expected S/N of the
                     visible part.

Everything with gaussian radiometer noise of CHARTS (SEFD of 256 antennas,
n_pol=1, engine/charts_sefd.py), without RFI, quantized to 8 bits with the
system noise at FLOOR_LEVEL and sigma = LEVELS_PER_SIGMA levels (see
write()). It is the scale agreed for the base dataset to which RFI is later
added as power (it raises the mean): it leaves ~56 sigma of headroom above
and 8 sigma below.

Physical choices (on purpose):
  - Dispersion smearing WITHIN each channel (0.3 MHz): fitburst does not
    apply it in SpectrumModeler (compute_time_dm_smear exists but is not
    used). In CHARTS it is not negligible: ~92 ms at 300 MHz with DM 1000,
    ~277 ms with DM 3000. It is applied as a boxcar of width
    |idx|·DM·K·Δν·f^(idx-1) per channel.
  - Spectrum normalized to the brightest channel of the band, and scattering
    with the normalized PBF (it conserves energy); see model().
  - Integration within the 1 ms sample (×4 oversampling), so that
    bursts of ~0.2 ms do not depend on the sampling phase.
  - Files of fixed length (NSAMP) for all sets, DM search
    20-3000 and DM drawn uniformly in 100-3000 (not the catalog
    distribution, which leaves ~14% above 1000). Width, scattering and
    spectrum are taken together from a random catalog FRB.

The per-channel profile uses the same fitburst routines as
SpectrumModeler.compute_model (compute_profile_pbf/gaussian, the running
power law of compute_spectrum_rpl, compute_time_dm_delay), but evaluated
only in a local window around the arrival in each channel.

No .npz are saved (they would be ~320 MB per file): each .fil is reproduced
from parametros_sorteados.csv + the seed derived from its name.

Output (--out, default outputs_comparacion/):
  config.json, parametros_sorteados.csv
  A_snr_fijo/*.fil + manifest.csv, B_amplitud_fija/..., negativos/..., parcial/...
The folder, file and column names are those of the existing dataset and are kept unchanged.

Usage:
  python 07_generate_comparison_set.py [--n 1000] [--n-partial 1000]
      [--snr-a 8] [--snr-b-median 10] [--snr-partial 15]
      [--workers 8] [--seed 20260923] [--out outputs_comparacion]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm
from fitburst.routines import ism as fb_ism
from fitburst.routines import profile as fb_profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

import charts_sefd  # noqa: E402
from fitburst_sim import DM_CONST_SECONDS, DM_INDEX, REF_FREQ, SCATTERING_INDEX  # noqa: E402

# write_fil from 03_ (same SIGPROC header and same fixed quantization as the outputs/ set)
_spec = importlib.util.spec_from_file_location("convert_to_fil", HERE / "03_convert_to_fil.py")
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)
write_fil = _conv.write_fil

# Quantization (agreed in the RFI model analysis): the system noise floor at
# level 32 and sigma = 4 levels. Headroom of (255-32)/4 ~ 56 sigma above, to
# later add RFI as power (it raises the mean) and for bright bursts in few
# channels; 8 sigma below. The quantization noise adds 1/(12·4²) = 0.5% of
# variance (~0.26% of S/N).
FLOOR_LEVEL = 32
LEVELS_PER_SIGMA = 4

CATALOGS_DIR = HERE / "catalogs"
EXCLUDE = {"FRB20200302A", "FRB20200427A"}  # invalid sp_idx/sp_run

# ---- Instrument (CHARTS band, 672 channels, 1 ms) ----
FREQ_LO = 300.0
NUM_FREQ = 672
FREQ_HI = FREQ_LO + NUM_FREQ * 0.3  # 501.6 MHz
FREQS = np.linspace(FREQ_LO, FREQ_HI, NUM_FREQ)
RES_FREQ_MHZ = (FREQ_HI - FREQ_LO) / (NUM_FREQ - 1)
TSAMP = 1e-3
SEFD_JY = charts_sefd.charts_sefd(256)
N_POL = charts_sefd.CHARTS_N_POL
SIGMA_CH = SEFD_JY / np.sqrt(N_POL * RES_FREQ_MHZ * 1e6 * TSAMP)  # Jy, the same for the 672 channels

# ---- Set design ----
DM_DRAW = (100.0, 3000.0)
DM_SEARCH = (20.0, 3000.0)
T_TOP_RANGE = (5.0, 15.0)   # arrival at the highest channel (501.6 MHz), seconds from the start of the file
GUARD_S = 10.0              # same margin as PAD_GUARD_SAMPLES in 03_ (prepsubband rounding)
DM_INDEX_MAG_LOW = (1.5, 1.8)
DM_INDEX_MAG_HIGH = (2.2, 2.5)
WIDTH_MIN_S = 1e-4          # with ×4 oversampling (0.25 ms), sampling error <5% for sigma >= 0.1 ms
OVERSAMPLING_T = 4
BOXCAR_MAX = 4096           # samples; equal to Heimdall's default maximum boxcar
FRAC_VISIBLE_RANGE = (0.05, 0.95)  # partial subset: fraction of the burst energy inside the file
# Catalog spectra with fewer channels than this above 50% of the peak in the
# band are discarded. Set to 0 ON PURPOSE (CHARTS team, 2026-09-24): the
# extreme fits (sp_idx ~1e3) that put all the energy in 1-4 channels saturate
# the 8-bit scale (up to 47% signal loss at S/N 8) and we want to see and
# understand them. They are identified in the manifest by n_canales_emision and
# perdida_recorte_pct.
MIN_EMISSION_CHANNELS = 0


def delay_s(dm, idx, f):
    """Delay relative to REF_FREQ, same routine fitburst uses."""
    return fb_ism.compute_time_dm_delay(dm, DM_CONST_SECONDS, idx, np.asarray(f, dtype=float), freq2=REF_FREQ)


def sweep_s(dm, idx):
    return float(delay_s(dm, idx, FREQ_LO) - delay_s(dm, idx, FREQ_HI))


def smear_s(dm, idx, f):
    """Smearing within one channel: |d delay/df| · Δν."""
    return abs(idx) * dm * DM_CONST_SECONDS * RES_FREQ_MHZ * np.asarray(f, dtype=float) ** (idx - 1)


NSAMP = int(np.ceil((T_TOP_RANGE[1] + sweep_s(DM_SEARCH[1], DM_INDEX) + GUARD_S + 5.0) / 1000 / TSAMP)) * 1000


def seed_for(base, name):
    return int(hashlib.sha256(f"{base}:{name}".encode()).hexdigest()[:8], 16)


# ============================================================================
# Catalog: pool of shapes (width, scattering, spectrum) of real FRBs
# ============================================================================
def _float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def n_emission_channels(sp_idx, sp_run):
    """Channels of the band with the spectrum above 50% of its maximum in the band."""
    x = np.log(FREQS / REF_FREQ)
    l = sp_idx * x + sp_run * x ** 2
    return int((l - l.max() >= np.log(0.5)).sum())


def load_pool():
    """Width/scattering/spectrum from catalog1 + catalog2 (spectrum with a peak, sp_run<0;
    no excluded/sidelobe events), without a cut on predicted S/N: here the amplitude
    is set by the design, not by the catalog."""
    pool = {}
    with open(CATALOGS_DIR / "catalog1_clean.csv") as f:
        for r in csv.DictReader(f):
            w, s = _float(r["width_fitb_s"]), _float(r["scat_s"])
            idx, run = _float(r["sp_idx"]), _float(r["sp_run"])
            if None in (w, idx, run) or run >= 0 or w < WIDTH_MIN_S or r["name"] in EXCLUDE:
                continue
            pool[r["name"]] = ("catalog1", w, max(s or 0.0, 0.0), idx, run)
    with open(CATALOGS_DIR / "chimefrbcat2.csv") as f:
        for r in csv.DictReader(f):
            if r["excluded_flag"] != "0" or r["sidelobe_flag"] != "0" or r["tns_name"] in EXCLUDE:
                continue
            w, s = _float(r["width_fitb"]), _float(r["scat_time"])
            idx, run = _float(r["sp_idx"]), _float(r["sp_run"])
            if None in (w, idx, run) or run >= 0 or w < WIDTH_MIN_S:
                continue
            pool[r["tns_name"]] = ("catalog2", w, max(s or 0.0, 0.0), idx, run)  # catalog2 overrides catalog1; catalog scat<0 -> 0
    return sorted((k,) + v for k, v in pool.items() if n_emission_channels(v[3], v[4]) >= MIN_EMISSION_CHANNELS)


def draw(n, rng, pool, prefix):
    rows = []
    for i in range(n):
        name, cat, w, s600, sp_idx, sp_run = pool[rng.integers(len(pool))]
        rows.append({
            "id": f"{prefix}{i:03d}",
            "forma_origen": name, "catalogo_origen": cat,
            "dm": float(rng.uniform(*DM_DRAW)),
            "width_s": w,
            "scat_600_s": s600,
            # same 600 MHz -> REF_FREQ conversion as engine/fitburst_sim.py
            "scat_ref_s": s600 * (600.0 / REF_FREQ) ** (-SCATTERING_INDEX),
            "sp_idx": sp_idx, "sp_run": sp_run,
            "n_canales_emision": n_emission_channels(sp_idx, sp_run),
            "t_top_s": float(rng.uniform(*T_TOP_RANGE)),
        })
    return rows


def draw_dm_index(rng):
    rng_range = DM_INDEX_MAG_LOW if rng.random() < 0.5 else DM_INDEX_MAG_HIGH
    return -float(rng.uniform(*rng_range))


# ============================================================================
# Per-channel model
# ============================================================================
def _moving_average(x, m):
    """Centred boxcar of m samples that conserves the sum."""
    if m <= 1:
        return x
    xp = np.concatenate([np.zeros(m // 2), x, np.zeros(m - 1 - m // 2)])
    c = np.concatenate([[0.0], np.cumsum(xp)])
    return (c[m:] - c[:-m]) / m


def model(dm, idx, width_s, scat_ref_s, sp_idx, sp_run, t_top_s):
    """Signal of unit amplitude (1 Jy intrinsic peak in the brightest
    channel), channel by channel.

    Returns (channels, t_c): channels[c] = (i0, values) in absolute
    samples of the file (i0 can be <0 and the end can go past NSAMP,
    for partial bursts); t_c = arrival in each channel, in seconds.
    """
    arrival_ref = t_top_s - float(delay_s(dm, idx, FREQS[-1]))
    t_c = arrival_ref + delay_s(dm, idx, FREQS)
    # Same running power law as fb_spectrum.compute_spectrum_rpl, but
    # normalized to the brightest channel of the band (not to REF_FREQ): with
    # narrow catalog spectra (e.g. sp_idx=1134, sp_run=-1090) the value at
    # REF_FREQ differs from the peak by factors of up to ~1e85.
    x = np.log(FREQS / REF_FREQ)
    log_spec = sp_idx * x + sp_run * x ** 2
    spectrum = np.exp(log_spec - log_spec.max())
    smear = smear_s(dm, idx, FREQS)
    tau = scat_ref_s * (FREQS / REF_FREQ) ** SCATTERING_INDEX
    U = OVERSAMPLING_T
    channels = []
    for c in range(NUM_FREQ):
        tail = min(10.0 * tau[c], 200.0)
        i0 = int(np.floor((t_c[c] - 6 * width_s - smear[c] / 2) / TSAMP)) - 1
        i1 = int(np.ceil((t_c[c] + 6 * width_s + tail + smear[c] / 2) / TSAMP)) + 2
        # sample k integrates [k·T - T/2, k·T + T/2) (centred, like fitburst's point sampling)
        tt = (i0 + (np.arange((i1 - i0) * U) + 0.5) / U - 0.5) * TSAMP
        rel = tt - t_c[c]
        if tau[c] > 0.0:
            rel = np.maximum(rel, -5 * width_s)  # same clipping as model.compute_profile
            # normalize=True: gaussian (peak 1) convolved with an
            # exponential of area 1 -> conserves energy. With fitburst's
            # default (normalize_pbf: False) the energy grows with tau and
            # the spectrum tilts as (f/REF_FREQ)^4.
            prof = fb_profile.compute_profile_pbf(rel, 0.0, width_s, np.asarray(FREQS[c]), REF_FREQ,
                                                  scat_ref_s, sc_index=SCATTERING_INDEX, normalize=True)
        else:
            prof = fb_profile.compute_profile_gaussian(rel, 0.0, width_s)
        prof = _moving_average(prof, int(round(smear[c] / TSAMP * U)))
        vals = prof.reshape(-1, U).mean(axis=1) * spectrum[c]
        channels.append((i0, vals))
    return channels, t_c


def dedispersed_series(channels, t_c, t_top_s, visible_only=False):
    """Sum of channels aligned at the true DM with integer shifts (like the
    pipelines). visible_only=True drops what falls outside [0, NSAMP)."""
    shift = np.round((t_c - t_top_s) / TSAMP).astype(int)
    start = min(i0 - s for (i0, _), s in zip(channels, shift))
    end = max(i0 + len(v) - s for (i0, v), s in zip(channels, shift))
    out = np.zeros(end - start + BOXCAR_MAX)
    for (i0, v), s in zip(channels, shift):
        if visible_only:
            k = i0 + np.arange(len(v))
            v = np.where((k >= 0) & (k < NSAMP), v, 0.0)
        a = i0 - s - start
        out[a:a + len(v)] += v
    return out


WIDTHS = np.unique(np.concatenate([np.arange(1, 65), np.round(np.geomspace(64, BOXCAR_MAX, 60)).astype(int)]))


def optimal_boxcar_snr(series):
    """Expected S/N (noise-free) with the best boxcar in WIDTHS; the noise of the
    sum of the channels is SIGMA_CH·sqrt(NUM_FREQ·w). Returns (snr, w_samples)."""
    c = np.concatenate([[0.0], np.cumsum(series)])
    best, w_best = 0.0, 1
    for w in WIDTHS:
        v = (c[w:] - c[:-w]).max() / np.sqrt(w)
        if v > best:
            best, w_best = v, int(w)
    return best / (SIGMA_CH * np.sqrt(NUM_FREQ)), w_best


def energy(channels):
    """(total energy per channel, energy inside the file per channel)."""
    tot = np.zeros(NUM_FREQ)
    inside = np.zeros(NUM_FREQ)
    for c, (i0, v) in enumerate(channels):
        tot[c] = v.sum()
        a, b = max(0, -i0), min(len(v), NSAMP - i0)
        inside[c] = v[a:b].sum() if b > a else 0.0
    return tot, inside


# ============================================================================
# Writing
# ============================================================================
def write(name, channels, scale_jy, out_path, seed_base):
    rng = np.random.default_rng(seed_for(seed_base, name))
    data = rng.standard_normal((NUM_FREQ, NSAMP), dtype=np.float32)
    data *= SIGMA_CH
    for c, (i0, v) in enumerate(channels):
        a, b = max(0, -i0), min(len(v), NSAMP - i0)
        if b > a:
            data[c, i0 + a:i0 + b] += (scale_jy * v[a:b]).astype(np.float32)
    # counts = floor(FLOOR + 0.5 + LEVELS·x/sigma) clipped to 0-255, i.e.
    # rounding to the nearest level. write_fil maps [lo, hi] -> [0, 255] with
    # floor, so it is given lo = -(FLOOR+0.5)/LEVELS·sigma and hi - lo = 255/LEVELS·sigma.
    lo_s = -(FLOOR_LEVEL + 0.5) / LEVELS_PER_SIGMA   # in sigma (-8.125)
    hi_s = lo_s + 255.0 / LEVELS_PER_SIGMA           # in sigma (+55.625)
    n_clip = int((data < lo_s * SIGMA_CH).sum() + (data >= hi_s * SIGMA_CH).sum())
    # Expected signal loss from clipping (expected value over the noise,
    # not the realization): sum(s - (E[clip(s+n)] - E[clip(n)])) / sum(s)
    s = np.concatenate([v for _, v in channels]) * (scale_jy / SIGMA_CH)
    s = s[s > 0.05]
    e0 = _clipped_mean(np.zeros(1), lo_s, hi_s)[0]
    loss = float((s - (_clipped_mean(s, lo_s, hi_s) - e0)).sum() / s.sum()) if s.size else 0.0
    centre = (lo_s + hi_s) / 2 * SIGMA_CH
    write_fil(data, FREQS, np.array([0.0, TSAMP]), out_path, nbits=8, source_name=name,
              fixed_scale=(centre, (hi_s - lo_s) / 2 * SIGMA_CH))
    return n_clip, 100.0 * loss


def _clipped_mean(s, lo, hi):
    """E[clip(s + n, lo, hi)] with n ~ N(0, 1), everything in units of sigma."""
    a, b = lo - s, hi - s
    return (s * (norm.cdf(b) - norm.cdf(a)) + norm.pdf(a) - norm.pdf(b)
            + lo * norm.cdf(a) + hi * norm.sf(b))


def _ctx(p, idx, dm=None):
    return model(p["dm"] if dm is None else dm, idx, p["width_s"], p["scat_ref_s"],
                 p["sp_idx"], p["sp_run"], p["t_top_s"])


def task_unit_snr(p):
    channels, t_c = _ctx(p, DM_INDEX)
    snr, w = optimal_boxcar_snr(dedispersed_series(channels, t_c, p["t_top_s"]))
    return p["id"], snr, w


def task_pair_AB(args):
    """A_i and B_i share the model: only the scale changes."""
    p, scale_a, scale_b, snr_u, w_opt, out, seed_base = args
    channels, _ = _ctx(p, DM_INDEX)
    rows = []
    for set_, scale, folder in (("A", scale_a, "A_snr_fijo"), ("B", scale_b, "B_amplitud_fija")):
        name = f"{set_}_{p['id']}"
        path = out / folder / f"{name}.fil"
        n_clip, loss = write(name, channels, scale, path, seed_base)
        rows.append((folder, {
            "name": name, "fil_path": str(path), "set": set_, **p, "dm_index": DM_INDEX,
            "escala_jy": scale, "snr_esperado": scale * snr_u, "boxcar_opt_ms": w_opt * TSAMP * 1e3,
            "barrido_s": sweep_s(p["dm"], DM_INDEX), "n_muestras_clip": n_clip, "perdida_recorte_pct": loss,
        }))
    return rows


def task_negative(args):
    p, idx, scale, out, seed_base = args
    dm_neg = p["dm"] * sweep_s(1.0, DM_INDEX) / sweep_s(1.0, idx)
    channels, _ = _ctx(p, idx, dm=dm_neg)
    name = f"NEG_{p['id']}"
    path = out / "negativos" / f"{name}.fil"
    n_clip, loss = write(name, channels, scale, path, seed_base)
    return [("negativos", {
        "name": name, "fil_path": str(path), "set": "NEG", **p,
        "dm_equivalente_pos": p["dm"], "dm": dm_neg, "dm_index": idx,
        "escala_jy": scale, "par_de": f"A_{p['id']}",
        "barrido_s": sweep_s(dm_neg, idx), "n_muestras_clip": n_clip, "perdida_recorte_pct": loss,
    })]


def place_cut(p):
    """t_top that leaves ~p['frac_objetivo'] of the energy inside the file.

    With the model at t_top=0, each channel has its energy E_c and its temporal
    centroid; when t_top moves, the channel is inside if its centroid falls in
    [0, NSAMP). "inicio" (start) keeps the channels that arrive last (low part of
    the band), "fin" (end) those that arrive first (high part). The exact fraction
    is stored in frac_energia_visible of the manifest."""
    channels, _ = model(p["dm"], DM_INDEX, p["width_s"], p["scat_ref_s"], p["sp_idx"], p["sp_run"], 0.0)
    E = np.array([v.sum() for _, v in channels])
    cen = np.array([(i0 + np.dot(np.arange(len(v)), v) / e) * TSAMP if e > 0 else np.nan
                    for (i0, v), e in zip(channels, E)])
    ok = np.isfinite(cen)
    E, cen = E[ok], cen[ok]
    order = np.argsort(-cen if p["borde"] == "inicio" else cen)
    j = min(int(np.searchsorted(np.cumsum(E[order]) / E.sum(), p["frac_objetivo"])), order.size - 1)
    c = cen[order[j]]
    return float(-c + 0.5 * TSAMP if p["borde"] == "inicio" else NSAMP * TSAMP - c - 0.5 * TSAMP)


def task_partial(args):
    p, snr_full, out, seed_base = args
    p = {**p, "t_top_s": place_cut(p)}
    channels, t_c = _ctx(p, DM_INDEX)
    snr_u, w_u = optimal_boxcar_snr(dedispersed_series(channels, t_c, p["t_top_s"]))
    snr_vis_u, w_vis = optimal_boxcar_snr(dedispersed_series(channels, t_c, p["t_top_s"], visible_only=True))
    tot, inside = energy(channels)
    frac_channel = np.divide(inside, tot, out=np.zeros_like(tot), where=tot > 0)
    visible = FREQS[frac_channel >= 0.5]
    scale = snr_full / snr_u
    name = f"P_{p['id']}_{p['borde']}"
    path = out / "parcial" / f"{name}.fil"
    n_clip, loss = write(name, channels, scale, path, seed_base)
    return [("parcial", {
        "name": name, "fil_path": str(path), "set": "P", **p, "dm_index": DM_INDEX,
        "escala_jy": scale, "snr_burst_completo": snr_full,
        "snr_visible_esperado": scale * snr_vis_u, "boxcar_opt_visible_ms": w_vis * TSAMP * 1e3,
        "frac_energia_visible": float(inside.sum() / tot.sum()) if tot.sum() > 0 else 0.0,
        "f_min_visible_mhz": float(visible.min()) if visible.size else np.nan,
        "f_max_visible_mhz": float(visible.max()) if visible.size else np.nan,
        "n_canales_visibles": int(visible.size),
        "barrido_s": sweep_s(p["dm"], DM_INDEX), "n_muestras_clip": n_clip, "perdida_recorte_pct": loss,
    })]


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for f in rows for k in f))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=1000, help="Draws for A, B and negatives (each)")
    ap.add_argument("--n-partial", type=int, default=1000)
    ap.add_argument("--snr-a", type=float, default=8.0)
    ap.add_argument("--snr-b-median", type=float, default=10.0,
                    help="S0 of B is calibrated so that the median expected S/N is this value")
    ap.add_argument("--snr-partial", type=float, default=15.0,
                    help="S/N of the FULL burst (before the cut) in the partial subset")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", type=Path, default=HERE / "outputs_comparacion")
    args = ap.parse_args()

    out = args.out.resolve()
    if out == (HERE / "outputs").resolve():
        sys.exit("✗ --out cannot be outputs/ (reference dataset, must not be modified)")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    pool = load_pool()
    main_draws = draw(args.n, rng, pool, "")
    idx_neg = [draw_dm_index(rng) for _ in main_draws]
    partials = draw(args.n_partial, rng, pool, "p")
    for j, p in enumerate(partials):
        # inicio: the high channel arrived before the file; fin: the low channel arrives after.
        # The visible energy fraction is drawn, not the time: with catalog spectra
        # concentrated at the top of the band (38% peak at 501.6 MHz), a cut uniform
        # in time leaves almost all "inicio" at ~0% and all "fin" at ~100%.
        # t_top is set in place_cut() and stored in the manifest.
        p["borde"] = "inicio" if j % 2 == 0 else "fin"
        p["frac_objetivo"] = float(rng.uniform(*FRAC_VISIBLE_RANGE))
        p["t_top_s"] = np.nan
    save_csv(out / "parametros_sorteados.csv",
             [{**p, "dm_index_negativo": i} for p, i in zip(main_draws, idx_neg)] + partials)
    print(f"Catalog pool: {len(pool)} shapes | NSAMP={NSAMP} ({NSAMP*TSAMP:.0f} s) | "
          f"SEFD={SEFD_JY:.2f} Jy | sigma_channel={SIGMA_CH:.3f} Jy", flush=True)

    with Pool(args.workers) as pool_proc:
        print(f"[phase 1] unit S/N of {len(main_draws)} draws...", flush=True)
        snr_u = {pid: (s, w) for pid, s, w in pool_proc.imap_unordered(task_unit_snr, main_draws)}
        snr_arr = np.array([snr_u[p["id"]][0] for p in main_draws])
        s0_b = args.snr_b_median / float(np.median(snr_arr))
        print(f"  S/N per Jy of amplitude: median {np.median(snr_arr):.4g}, "
              f"range {snr_arr.min():.3g}-{snr_arr.max():.3g} | S0 (B) = {s0_b:.4g} Jy", flush=True)

        tasks = []
        for p, idx in zip(main_draws, idx_neg):
            s, w = snr_u[p["id"]]
            scale_a = args.snr_a / s
            tasks.append((task_pair_AB, (p, scale_a, s0_b, s, w, out, args.seed)))
            tasks.append((task_negative, (p, idx, scale_a, out, args.seed)))
        tasks += [(task_partial, (p, args.snr_partial, out, args.seed)) for p in partials]

        print(f"[phase 2] writing {2 * len(main_draws) + len(main_draws) + len(partials)} .fil "
              f"with {args.workers} workers...", flush=True)
        manifests = {}
        done = 0
        for rows in pool_proc.imap_unordered(_dispatch, tasks):
            for folder, row in rows:
                manifests.setdefault(folder, []).append(row)
                done += 1
            if done % 50 < len(rows):
                print(f"  {done} files ({(time.time() - t0) / 60:.1f} min)", flush=True)

    for folder, rows in manifests.items():
        rows.sort(key=lambda f: f["name"])
        save_csv(out / folder / "manifest.csv", rows)

    # config.json keys are those of the existing dataset (kept unchanged)
    config = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "script": Path(__file__).name, **vars(args),
        "out": str(out), "S0_B_jy": s0_b, "nsamp": NSAMP, "tsamp_s": TSAMP, "nbits": 8,
        "freq_lo_mhz": FREQ_LO, "freq_hi_mhz": FREQ_HI, "num_freq": NUM_FREQ,
        "sefd_jy": SEFD_JY, "n_pol": N_POL, "sigma_canal_jy": float(SIGMA_CH),
        "cuantizacion": f"uint8, system noise at {FLOOR_LEVEL} with sigma = {LEVELS_PER_SIGMA} levels, rounding",
        "piso_niveles": FLOOR_LEVEL, "niveles_por_sigma": LEVELS_PER_SIGMA,
        "dm_sorteo": DM_DRAW, "dm_search": DM_SEARCH, "t_top_rango_s": T_TOP_RANGE,
        "sobremuestreo_t": OVERSAMPLING_T, "boxcar_max_muestras": BOXCAR_MAX,
        "smearing_intracanal": True, "rfi": False, "pool_catalogo": len(pool), "min_canales_emision": MIN_EMISSION_CHANNELS,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"✓ Done in {(time.time() - t0) / 60:.1f} min -> {out}")


def _dispatch(t):
    fn, a = t
    return fn(a)


if __name__ == "__main__":
    main()
