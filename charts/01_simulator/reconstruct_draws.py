#!/usr/bin/env python3
"""Rebuilds parametros_sorteados.csv (the group-2 draws that group 3 reuses) from the group-3 manifest.csv files.
The manifests store every column read by 12_generate_comparison_set_21504.py; t_top_s is not used
(the generator recomputes the arrival). The column names are those of the existing dataset.

Usage: python reconstruct_draws.py GROUP3_FOLDER OUTPUT.csv
"""
import csv
import sys
from pathlib import Path

base, output = Path(sys.argv[1]), Path(sys.argv[2])
common = ["id", "forma_origen", "catalogo_origen", "dm", "width_s", "scat_600_s", "scat_ref_s", "sp_idx", "sp_run",
          "n_canales_emision", "t_top_s"]
rows = []
for r in csv.DictReader(open(base / "A_snr_fijo" / "manifest.csv")):
    rows.append({**{k: r[k] for k in common}, "dm_index_negativo": r["dm_index_negativo"], "borde": "", "frac_objetivo": ""})
for r in csv.DictReader(open(base / "parcial" / "manifest.csv")):
    rows.append({**{k: r[k] for k in common}, "dm_index_negativo": "", "borde": r["borde"], "frac_objetivo": r["frac_objetivo"]})
with open(output, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} draws -> {output}")
