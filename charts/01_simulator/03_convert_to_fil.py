#!/usr/bin/env python3
"""
SIGPROC .fil writer (write_fil) used by the group-2 and group-3 generators, and converter of the
group-1 .npz files (positives and negatives) to .fil, ready for Heimdall/PRESTO/AstroAccelerate.

Writes the SIGPROC header by hand (length-prefixed labels) -- it does not depend on sigpyproc.
For the .npz conversion it pads with gaussian noise (robust per-channel statistics, median/MAD) on
both sides of the burst, so that a blind DM search up to DM_SEARCH_MAX does not lose it at the edges.

The .npz inputs come from the group-1 generators (not included in this repository). The input and
output file and column names are those of the existing group-1 dataset and are kept unchanged.

Output (output folder name = input folder name + "_fil", e.g. positivos -> positivos_fil):
  outputs/<positives-dir>_fil/<catalogo>/*.fil, .../manifest.csv
  outputs/<negatives-dir>_fil/*.fil,            .../manifest.csv

Usage:
  python 03_convert_to_fil.py [--limit N] [--nbits 8] [--skip-negatives]
                              [--positives-dir outputs/positivos]
                              [--negatives-dir outputs/negativos]
"""
import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from charts_sefd import theoretical_noise_std  # noqa: E402

N_SIGMA_CLIP = 5.0  # in units of the theoretical CHARTS noise sigma

# Fixed DM search window for the detection tools (group 1: "DM from 20 to 1000
# for all"). PAD_GUARD_SAMPLES=10000 because prepsubband (PRESTO) rounds its
# output length up (FFT-friendly) before computing how many samples are valid --
# with a small guard that can empty the whole file for bursts with a short
# window (confirmed in 27/2839 files).
DM_SEARCH_MIN = 20.0
DM_SEARCH_MAX = 1000.0
PAD_GUARD_SAMPLES = 10000


def write_fil(data_full, freqs, times, out_path, nbits=8, source_name="FRB", fixed_scale=None):
    data = np.asarray(data_full, dtype=np.float64)

    if nbits in (8, 16) and fixed_scale is not None:
        # FIXED scale, anchored to the THEORETICAL system noise sigma
        # (radiometer equation, see engine/charts_sefd.py::theoretical_noise_std),
        # not the GLOBAL minimum/maximum of this particular file. With a
        # per-file minimum/maximum the burst peak dominates the range and
        # squeezes the noise floor to a different fraction of 0-255 in each
        # file depending on how bright its burst is (confirmed 2026-08-30:
        # floor at 130/255 for S/N=0.03 vs 9/255 for S/N=6.20, same instrument,
        # same physical noise). With a fixed scale the floor looks the same in
        # every file.
        centre, half_width = fixed_scale
        lo, hi = centre - half_width, centre + half_width
        norm01 = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        if nbits == 8:
            norm = (norm01 * 255).astype(np.uint8)
        else:
            norm = (norm01 * 65535).astype(np.uint16)
    elif nbits == 8:
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        drange = dmax - dmin if dmax > dmin else 1.0
        norm = ((data - dmin) / drange * 255).astype(np.uint8)
    elif nbits == 16:
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        drange = dmax - dmin if dmax > dmin else 1.0
        norm = ((data - dmin) / drange * 65535).astype(np.uint16)
    elif nbits == 32:
        norm = data.astype(np.float32)
    else:
        raise ValueError(f"nbits={nbits} not supported")

    # SIGPROC: descending frequencies, rows = time samples
    data_fil = norm[::-1, :].T.copy()

    tsamp = float(times[1] - times[0])
    fch1 = float(freqs[-1])
    foff = float(freqs[0] - freqs[-1]) / (len(freqs) - 1)

    def wraw_str(buf, s):
        b = s.encode("ascii")
        return buf + struct.pack("<i", len(b)) + b

    def wlabel_str(buf, label, value):
        buf = wraw_str(buf, label)
        return wraw_str(buf, value)

    def wlabel_num(buf, label, value, fmt):
        buf = wraw_str(buf, label)
        return buf + struct.pack(fmt, value)

    hdr = wraw_str(b"", "HEADER_START")
    hdr = wlabel_str(hdr, "source_name", source_name[:80])
    hdr = wlabel_num(hdr, "machine_id", 0, "<i")
    hdr = wlabel_num(hdr, "telescope_id", 0, "<i")
    hdr = wlabel_num(hdr, "data_type", 1, "<i")
    hdr = wlabel_num(hdr, "nbits", nbits, "<i")
    hdr = wlabel_num(hdr, "nchans", len(freqs), "<i")
    hdr = wlabel_num(hdr, "fch1", fch1, "<d")
    hdr = wlabel_num(hdr, "foff", foff, "<d")
    hdr = wlabel_num(hdr, "nifs", 1, "<i")
    hdr = wlabel_num(hdr, "tstart", 58000.0, "<d")
    hdr = wlabel_num(hdr, "tsamp", tsamp, "<d")
    hdr = wraw_str(hdr, "HEADER_END")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(hdr)
        f.write(data_fil.tobytes())
    return out_path.stat().st_size


def dispersion_delay_s(dm, freq_lo_mhz, freq_hi_mhz, dm_const_seconds):
    return dm_const_seconds * dm * (1.0 / freq_lo_mhz ** 2 - 1.0 / freq_hi_mhz ** 2)


def compute_pad_samples(tsamp, freq_lo_mhz, freq_hi_mhz, dm_const_seconds):
    delay_max = dispersion_delay_s(DM_SEARCH_MAX, freq_lo_mhz, freq_hi_mhz, dm_const_seconds)
    return int(np.ceil(delay_max / tsamp)) + PAD_GUARD_SAMPLES


def pad_with_noise(data_full, tsamp, freq_lo_mhz, freq_hi_mhz, dm_const_seconds):
    pad_samples = compute_pad_samples(tsamp, freq_lo_mhz, freq_hi_mhz, dm_const_seconds)

    nchan = data_full.shape[0]
    med = np.median(data_full, axis=1, keepdims=True)
    mad = np.median(np.abs(data_full - med), axis=1, keepdims=True)
    sigma = np.where(mad > 0, mad * 1.4826, np.std(data_full, axis=1, keepdims=True))
    sigma = np.where(sigma > 0, sigma, 1.0)

    rng = np.random.default_rng()
    noise_before = med + sigma * rng.standard_normal((nchan, pad_samples))
    noise_after = med + sigma * rng.standard_normal((nchan, pad_samples))

    data_padded = np.concatenate([noise_before, data_full, noise_after], axis=1).astype(data_full.dtype)
    return data_padded, pad_samples


def _scalar(bp, key, default=None):
    v = bp.get(key, default)
    if isinstance(v, (list, np.ndarray)):
        return float(v[0]) if len(v) else default
    return v


def load_meta(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = d["metadata"]
    if hasattr(meta, "item"):
        meta = meta.item()
    bp = d["burst_parameters"]
    if hasattr(bp, "item"):
        bp = bp.item()

    num_freq = int(meta["num_freq"])
    freqs = meta["freqs_bin0"] + meta["res_freq"] * np.arange(num_freq)
    tsamp = float(meta["res_time"])
    dm_const_seconds = float(meta["dm_const_seconds"])

    pad_before = compute_pad_samples(tsamp, float(freqs[0]), float(freqs[-1]), dm_const_seconds)
    arrival_time_s = _scalar(bp, "arrival_time")
    arrival_time_in_fil_s = arrival_time_s + pad_before * tsamp if arrival_time_s is not None else None

    return {
        "dm_truth": _scalar(bp, "dm"),
        "arrival_time_in_fil_s": arrival_time_in_fil_s,
        "pad_samples_before": pad_before,
        "burst_width_s": _scalar(bp, "burst_width"),
        "snr_peak_truth": _scalar(bp, "snr_peak"),
        "snr_integrated_truth": _scalar(bp, "snr_integrated"),
        "num_freq": num_freq,
        "res_time_s": tsamp,
    }, meta, bp


def convert_one(npz_path, out_path, nbits, name):
    d = np.load(npz_path, allow_pickle=True)
    data_full = np.asarray(d["data_full"], dtype=np.float64)
    info, meta, bp = load_meta(npz_path)

    num_freq = info["num_freq"]
    freqs = meta["freqs_bin0"] + meta["res_freq"] * np.arange(num_freq)
    tsamp = info["res_time_s"]
    dm_const_seconds = float(meta["dm_const_seconds"])

    data_padded, pad_before = pad_with_noise(
        data_full, tsamp, float(freqs[0]), float(freqs[-1]), dm_const_seconds
    )
    num_time_padded = data_padded.shape[1]
    times_padded = np.arange(num_time_padded) * tsamp

    sigma_theo = theoretical_noise_std(meta)
    fixed_scale = (0.0, N_SIGMA_CLIP * sigma_theo)  # no RFI: a single sigma, uniform across channels
    size = write_fil(data_padded, freqs, times_padded, out_path, nbits=nbits, source_name=name,
                     fixed_scale=fixed_scale)
    info["fil_size_bytes"] = size
    info["cuantizacion_sigma_teo_jy"] = sigma_theo
    info["cuantizacion_n_sigma"] = N_SIGMA_CLIP
    return info


def convert_positives(nbits, limit, src_root, out_dir):
    summary_csv = src_root / "frb_simulados_resumen.csv"
    manifest_csv = out_dir / "manifest.csv"
    if not summary_csv.exists():
        print(f"  (skipped: {summary_csv} does not exist)")
        return
    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_ok, n_err = 0, 0
    for i, row in enumerate(rows, 1):
        name = row["name"]
        catalog = row["catalogo"]
        npz_path = src_root / row["npz_path"]
        out_path = out_dir / catalog / f"{catalog}_{name}.fil"
        try:
            info = convert_one(npz_path, out_path, nbits, name)
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  [positives {i}/{len(rows)}] converted={n_ok} errors={n_err}", flush=True)
        except Exception as e:
            print(f"  [positives {i}/{len(rows)}] ERROR {name}: {e}", file=sys.stderr)
            n_err += 1
            continue
        manifest_rows.append({
            "name": name, "catalogo": catalog, "fil_path": str(out_path), "npz_path": str(npz_path),
            "dm_truth": info["dm_truth"], "arrival_time_in_fil_s": info["arrival_time_in_fil_s"],
            "pad_samples_before": info["pad_samples_before"], "burst_width_s": info["burst_width_s"],
            "snr_peak_truth": info["snr_peak_truth"], "snr_integrated_truth": info["snr_integrated_truth"],
            "res_time_s": info["res_time_s"], "dm_search_min": DM_SEARCH_MIN, "dm_search_max": DM_SEARCH_MAX,
            "cuantizacion_sigma_teo_jy": info["cuantizacion_sigma_teo_jy"],
            "cuantizacion_n_sigma": info["cuantizacion_n_sigma"],
        })

    if manifest_rows:
        with open(manifest_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
    print(f"✓ Positives: {n_ok} converted, {n_err} errors -> {manifest_csv}")


def convert_negatives(nbits, limit, src_root, out_dir):
    manifest_in = src_root / "manifest_negativos.csv"
    manifest_csv = out_dir / "manifest.csv"
    if not manifest_in.exists():
        print(f"  (skipped: {manifest_in} does not exist; it is written by the group-1 negatives generator)")
        return
    with open(manifest_in) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_ok, n_err = 0, 0
    for i, row in enumerate(rows, 1):
        name = row["name"]
        npz_path = src_root / row["npz_path"]
        out_path = out_dir / f"{name}.fil"
        try:
            info = convert_one(npz_path, out_path, nbits, name)
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  [negatives {i}/{len(rows)}] converted={n_ok} errors={n_err}", flush=True)
        except Exception as e:
            print(f"  [negatives {i}/{len(rows)}] ERROR {name}: {e}", file=sys.stderr)
            n_err += 1
            continue
        manifest_rows.append({
            "name": name, "name_origen": row["name_origen"], "fil_path": str(out_path),
            "dm_truth": info["dm_truth"], "dm_index_usado": row["dm_index_usado"],
            "arrival_time_in_fil_s": info["arrival_time_in_fil_s"],
            "pad_samples_before": info["pad_samples_before"], "burst_width_s": info["burst_width_s"],
            "snr_peak_truth": info["snr_peak_truth"], "snr_integrated_truth": info["snr_integrated_truth"],
            "res_time_s": info["res_time_s"], "dm_search_min": DM_SEARCH_MIN, "dm_search_max": DM_SEARCH_MAX,
            "cuantizacion_sigma_teo_jy": info["cuantizacion_sigma_teo_jy"],
            "cuantizacion_n_sigma": info["cuantizacion_n_sigma"],
        })

    if manifest_rows:
        with open(manifest_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
    print(f"✓ Negatives: {n_ok} converted, {n_err} errors -> {manifest_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--skip-negatives", action="store_true")
    ap.add_argument("--positives-dir", type=Path, default=HERE / "outputs" / "positivos",
                    help="Folder with frb_simulados_resumen.csv (group-1 positives). "
                         "Default: outputs/positivos")
    ap.add_argument("--negatives-dir", type=Path, default=HERE / "outputs" / "negativos",
                    help="Folder with manifest_negativos.csv (group-1 negatives). "
                         "Default: outputs/negativos")
    args = ap.parse_args()

    pos_out_dir = args.positives_dir.parent / f"{args.positives_dir.name}_fil"
    neg_out_dir = args.negatives_dir.parent / f"{args.negatives_dir.name}_fil"

    print("Converting positives...")
    convert_positives(args.nbits, args.limit, args.positives_dir, pos_out_dir)

    if not args.skip_negatives:
        print("\nConverting negatives...")
        convert_negatives(args.nbits, args.limit, args.negatives_dir, neg_out_dir)


if __name__ == "__main__":
    main()
