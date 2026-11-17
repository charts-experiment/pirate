#!/usr/bin/env python3
"""
Group 3: comparison set at 21 504 channels (agreed with the CHARTS team 2026-10-05/08).

The SAME draws as group 2 (outputs_comparacion/, 07_generate_comparison_set.py, read from its
parametros_sorteados.csv, or from group2_drawn_parameters.csv), with the same physics (model() of 07_:
smearing within the channel, spectrum normalized to the brightest channel, energy-conserving
scattering, ×4 oversampling), on a fine grid:

  - 21 504 channels = each of the 672 channels (300.45 kHz) split in 32 (9.389 kHz).
    Centres from 299.855 to 501.745 MHz, sigma = SEFD / sqrt(9.389 kHz · 1 ms) = 768.5 Jy.
  - A: S/N 8 recomputed with the 21 504-channel model (optimal boxcar, dedispersed at the true DM).
  - B: S0 = 616.87 Jy, the SAME as in group 2 (the S/N difference comes only from the grid).
  - negativos: pair of A_i with its dm_index_negativo and the scale of A_i.
  - parcial: full burst at S/N 15 (recomputed), same edge and target energy fraction.
  - VARIABLE length, no padding: arrival at the 501.6 MHz channel drawn in 3-5 s
    (seed per id) + sweep of its own DM + pulse tail in the lowest channel + 1 s,
    rounded up to the second. In parcial, the same length and the cut of 07_.
  - t_top_s in the manifest = arrival at 501.6 MHz (same convention as group 2).
  - 8 bits, floor 32, 1 sigma = 4 levels, rounding (same quantization as write_fil in 03_).
The folder, file and column names are those of the existing dataset and are kept unchanged.

Written in time blocks (a 98 s file weighs ~2 GB in uint8 and 8.4 GB in float32).
Resumable: finished files are recorded in _done.jsonl and skipped (older runs: _hechos.jsonl is also read).

Usage (simulator venv):
  python 12_generate_comparison_set_21504.py [--limit 10] [--workers 24]
      [--out outputs_comparacion_21504] [--sets A,B,NEG,P]
"""
import argparse
import csv
import importlib.util
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gen07", HERE / "07_generate_comparison_set.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

BASE_G2 = HERE / "outputs_comparacion"          # group 2 (if present)
# group-2 draws: those of group 2 if present; otherwise the copy reconstructed from the group-3 manifests
DRAWS = BASE_G2 / "parametros_sorteados.csv"
if not DRAWS.exists():
    DRAWS = HERE / "group2_drawn_parameters.csv"
SEED = 20260923                                 # group-2 seed
S0_B_JY = 616.8725247769761                     # group-2 config.json
SNR_A, SNR_PARTIAL = 8.0, 15.0
T_TOP_ARRIVAL = (3.0, 5.0)
MARGIN_S = 1.0
F_REF_TOP = 501.6                               # t_top = arrival at 501.6 MHz
BLOCK = 4096                                    # samples per writing block

# ---- Fine grid: 672 channels x 32 ----
N_SUB = 32
COARSE = np.linspace(300.0, 501.6, 672)
DNU_COARSE = (501.6 - 300.0) / 671
DNU = DNU_COARSE / N_SUB
FREQS = (COARSE[:, None] + (np.arange(N_SUB) - (N_SUB - 1) / 2) * DNU).ravel()   # ascending
G.FREQS = FREQS
G.NUM_FREQ = FREQS.size
G.RES_FREQ_MHZ = DNU
G.SIGMA_CH = G.SEFD_JY / np.sqrt(G.N_POL * DNU * 1e6 * G.TSAMP)
SIGMA = G.SIGMA_CH
TSAMP = G.TSAMP


def model_t_top(dm, idx, t_top_501_6):
    """model() of 07_ uses t_top = arrival at the highest channel (FREQS[-1] = 501.745 MHz)."""
    return t_top_501_6 + float(G.delay_s(dm, idx, FREQS[-1]) - G.delay_s(dm, idx, F_REF_TOP))


def length_samples(dm, idx, t_top, p):
    tau_lo = p["scat_ref_s"] * (FREQS[0] / G.REF_FREQ) ** G.SCATTERING_INDEX
    tail = 6 * p["width_s"] + min(10 * tau_lo, 200.0) + float(G.smear_s(dm, idx, FREQS[0])) / 2
    end = t_top + G.sweep_s(dm, idx) + tail + MARGIN_S
    return int(math.ceil(end)) * 1000


def arrival_time(pid):
    return float(np.random.default_rng(G.seed_for(SEED, f"t_top_21504:{pid}")).uniform(*T_TOP_ARRIVAL))


# ---------------------------------------------------------------------------
def write(name, channels, scale_jy, path, nsamp):
    """Noise + signal in blocks, quantized like write_fil(fixed_scale) of 03_:
    level = floor(32.5 + 4·x/sigma) clipped to 0-255. SIGPROC order: descending channels."""
    tmp = path.with_suffix(".fil.tmp")
    G.write_fil(np.zeros((FREQS.size, 0)), FREQS, np.array([0.0, TSAMP]), tmp, nbits=8, source_name=name,
                fixed_scale=(0.0, 1.0))                     # header only (it does not depend on the scale)
    rng = np.random.default_rng(G.seed_for(SEED, name))
    i0s = np.array([i0 for i0, _ in channels])
    i1s = np.array([i0 + len(v) for i0, v in channels])
    n_clip = 0
    with open(tmp, "ab") as f:
        for a in range(0, nsamp, BLOCK):
            b = min(a + BLOCK, nsamp)
            x = rng.standard_normal((b - a, FREQS.size), dtype=np.float32)    # (time, ascending channel), in sigma
            for c in np.flatnonzero((i1s > a) & (i0s < b)):
                i0, v = channels[c]
                lo, hi = max(a, i0), min(b, i0 + len(v))
                x[lo - a:hi - a, c] += (scale_jy / SIGMA) * v[lo - i0:hi - i0]
            q = np.floor(x * G.LEVELS_PER_SIGMA + (G.FLOOR_LEVEL + 0.5))
            n_clip += int((q < 0).sum() + (q > 255).sum())
            f.write(np.clip(q, 0, 255).astype(np.uint8)[:, ::-1].tobytes())
    tmp.rename(path)
    return n_clip


def base(p, idx, t_top, nsamp, dm=None):
    G.NSAMP = nsamp                                       # dedispersed_series / energy / place_cut
    dm = p["dm"] if dm is None else dm
    channels, t_c = G.model(dm, idx, p["width_s"], p["scat_ref_s"], p["sp_idx"], p["sp_run"],
                            model_t_top(dm, idx, t_top))
    return channels, t_c


def task_main(args):
    """A_i, B_i and NEG_i (they share draw, arrival and length)."""
    p, out, sets = args
    t_top = arrival_time(p["id"])
    nsamp = length_samples(p["dm"], G.DM_INDEX, t_top, p)
    channels, t_c = base(p, G.DM_INDEX, t_top, nsamp)
    snr_u, w = G.optimal_boxcar_snr(G.dedispersed_series(channels, t_c, t_c[-1]))
    scale_a = SNR_A / snr_u
    common = {**p, "t_top_s": t_top, "nsamp": nsamp, "largo_s": nsamp * TSAMP, "snr_por_jy": snr_u}
    rows = []
    for set_, scale, folder in (("A", scale_a, "A_snr_fijo"), ("B", S0_B_JY, "B_amplitud_fija")):
        if set_ not in sets:
            continue
        name = f"{set_}_{p['id']}"
        path = out / folder / f"{name}.fil"
        n_clip = write(name, channels, scale, path, nsamp)
        rows.append((folder, {"name": name, "fil_path": str(path), "set": set_, **common,
                              "dm_index": G.DM_INDEX, "escala_jy": scale, "snr_esperado": scale * snr_u,
                              "boxcar_opt_ms": w * TSAMP * 1e3, "barrido_s": G.sweep_s(p["dm"], G.DM_INDEX),
                              "n_muestras_clip": n_clip}))
    if "NEG" in sets:
        idx = p["dm_index_negativo"]
        dm_neg = p["dm"] * G.sweep_s(1.0, G.DM_INDEX) / G.sweep_s(1.0, idx)
        channels_n, _ = base(p, idx, t_top, nsamp, dm=dm_neg)
        name = f"NEG_{p['id']}"
        path = out / "negativos" / f"{name}.fil"
        n_clip = write(name, channels_n, scale_a, path, nsamp)
        rows.append(("negativos", {"name": name, "fil_path": str(path), "set": "NEG", **common,
                                   "dm_equivalente_pos": p["dm"], "dm": dm_neg, "dm_index": idx,
                                   "escala_jy": scale_a, "par_de": f"A_{p['id']}",
                                   "barrido_s": G.sweep_s(dm_neg, idx), "n_muestras_clip": n_clip}))
    return rows


def task_partial(args):
    p, out, _ = args
    nsamp = length_samples(p["dm"], G.DM_INDEX, T_TOP_ARRIVAL[1], p)
    G.NSAMP = nsamp
    t_top_mod = G.place_cut(p)                            # arrival at the highest channel of the fine grid
    t_top = t_top_mod - float(G.delay_s(p["dm"], G.DM_INDEX, FREQS[-1]) - G.delay_s(p["dm"], G.DM_INDEX, F_REF_TOP))
    channels, t_c = base(p, G.DM_INDEX, t_top, nsamp)
    snr_u, _ = G.optimal_boxcar_snr(G.dedispersed_series(channels, t_c, t_c[-1]))
    snr_vis_u, w_vis = G.optimal_boxcar_snr(G.dedispersed_series(channels, t_c, t_c[-1], visible_only=True))
    tot, inside = G.energy(channels)
    frac_channel = np.divide(inside, tot, out=np.zeros_like(tot), where=tot > 0)
    visible = FREQS[frac_channel >= 0.5]
    scale = SNR_PARTIAL / snr_u
    name = f"P_{p['id']}_{p['borde']}"
    path = out / "parcial" / f"{name}.fil"
    n_clip = write(name, channels, scale, path, nsamp)
    return [("parcial", {"name": name, "fil_path": str(path), "set": "P", **p, "t_top_s": t_top,
                         "nsamp": nsamp, "largo_s": nsamp * TSAMP, "dm_index": G.DM_INDEX, "escala_jy": scale,
                         "snr_burst_completo": SNR_PARTIAL, "snr_visible_esperado": scale * snr_vis_u,
                         "boxcar_opt_visible_ms": w_vis * TSAMP * 1e3,
                         "frac_energia_visible": float(inside.sum() / tot.sum()) if tot.sum() > 0 else 0.0,
                         "f_min_visible_mhz": float(visible.min()) if visible.size else np.nan,
                         "f_max_visible_mhz": float(visible.max()) if visible.size else np.nan,
                         "n_canales_visibles": int(visible.size),
                         "barrido_s": G.sweep_s(p["dm"], G.DM_INDEX), "n_muestras_clip": n_clip})]


def _dispatch(t):
    fn, a = t
    t0 = time.time()
    rows = fn(a)
    return rows, time.time() - t0


def read_draws():
    num = {"dm", "width_s", "scat_600_s", "scat_ref_s", "sp_idx", "sp_run", "t_top_s",
           "dm_index_negativo", "frac_objetivo"}
    main_draws, partials = [], []
    with open(DRAWS, newline="") as f:
        for r in csv.DictReader(f):
            p = {k: (float(v) if k in num and v not in ("", None) else v) for k, v in r.items()}
            p["n_canales_emision"] = G.n_emission_channels(p["sp_idx"], p["sp_run"])   # on the fine grid
            (partials if p["id"].startswith("p") else main_draws).append(p)
    for p in main_draws:
        p.pop("borde", None), p.pop("frac_objetivo", None)
    for p in partials:
        p.pop("dm_index_negativo", None)
    return main_draws, partials


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=HERE / "outputs_comparacion_21504")
    ap.add_argument("--limit", type=int, default=None, help="first N draws of each kind (test)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sets", default="A,B,NEG,P")
    a = ap.parse_args()
    sets = set(a.sets.split(","))
    out = a.out.resolve()
    for d in ("A_snr_fijo", "B_amplitud_fija", "negativos", "parcial"):
        (out / d).mkdir(parents=True, exist_ok=True)
    done_path = out / "_done.jsonl"
    done = {}
    for path, k_key, k_rows in ((out / "_hechos.jsonl", "clave", "filas"), (done_path, "key", "rows")):
        if path.exists():                                 # _hechos.jsonl: runs before the translation
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done[r[k_key]] = r[k_rows]

    main_draws, partials = read_draws()
    if a.limit:
        main_draws, partials = main_draws[:a.limit], partials[:a.limit]
    tasks = []
    if sets & {"A", "B", "NEG"}:
        tasks += [(task_main, (p, out, sets)) for p in main_draws if f"main:{p['id']}" not in done]
    if "P" in sets:
        tasks += [(task_partial, (p, out, sets)) for p in partials if f"p:{p['id']}" not in done]
    print(f"Grid: {FREQS.size} channels, {FREQS[0]:.4f}-{FREQS[-1]:.4f} MHz, dnu={DNU * 1e3:.4f} kHz, "
          f"sigma={SIGMA:.2f} Jy | pending tasks {len(tasks)} ({len(done)} done) | {a.workers} workers",
          flush=True)

    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=20) as pool, open(done_path, "a") as fh:
        for n, (rows, sec) in enumerate(pool.imap_unordered(_dispatch, tasks), 1):
            key = ("main:" + rows[0][1]["name"].split("_", 1)[1]) if rows[0][0] != "parcial" \
                else "p:" + rows[0][1]["id"]
            fh.write(json.dumps({"key": key, "sec": sec, "rows": rows}, default=float) + "\n")
            fh.flush()
            done[key] = rows
            if n % 10 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  {n}/{len(tasks)} tasks, {el / 60:.1f} min, ETA {(len(tasks) - n) * el / n / 3600:.2f} h",
                      flush=True)

    manifests = {}
    for rows in done.values():
        for folder, row in rows:
            manifests.setdefault(folder, []).append(row)
    for folder, rows in manifests.items():
        rows.sort(key=lambda f: f["name"])
        G.save_csv(out / folder / "manifest.csv", rows)
    # config.json keys are those of the existing dataset (kept unchanged)
    (out / "config.json").write_text(json.dumps({
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "script": Path(__file__).name, "base_sorteos": str(BASE_G2),
        "seed": SEED, "num_freq": int(FREQS.size), "freq_min_mhz": float(FREQS[0]), "freq_max_mhz": float(FREQS[-1]),
        "dnu_khz": DNU * 1e3, "sigma_canal_jy": float(SIGMA), "sefd_jy": G.SEFD_JY, "tsamp_s": TSAMP, "nbits": 8,
        "piso_niveles": G.FLOOR_LEVEL, "niveles_por_sigma": G.LEVELS_PER_SIGMA, "snr_a": SNR_A,
        "S0_B_jy": S0_B_JY, "snr_parcial": SNR_PARTIAL, "t_top_llegada_s": T_TOP_ARRIVAL, "margen_s": MARGIN_S,
        "t_top_referencia_mhz": F_REF_TOP, "largo": "variable: ceil(t_top + sweep + pulse tail + 1 s)",
        "smearing_intracanal": True, "rfi": False, "limite": a.limit,
    }, indent=2))
    print(f"✓ {sum(len(v) for v in manifests.values())} files in {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
