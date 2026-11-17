"""Helpers shared by the 10_rfi_analysis scripts: locating runs and re-running the evaluator on a set of events.

Runs of the earlier Spanish-named chain (resultados/, asignacion.csv, emisor.json, eventos_todos.log, resultados.md,
mejor_snr_por_archivo.npy) are also read.
"""
import os
import subprocess
import sys
from pathlib import Path

if "CHARTS_RFI_RESULTS" not in os.environ:
    sys.exit("ERROR: export CHARTS_RFI_RESULTS=/folder/with/results/<run> (the runs to analyse)")
R = Path(os.environ["CHARTS_RFI_RESULTS"])
RUNS_DIR = R / "results" if (R / "results").exists() else R / "resultados"
PC = Path(__file__).resolve().parent.parent        # charts/ (09_evaluation/evaluate_test_64b.py)
PY = sys.executable                                # run with the Python of the pirate environment
K_TOP = 4.148808e3 * (299.8497764530551 ** -2 - 501.6 ** -2)   # s per unit DM, bottom band edge -> 501.6 MHz


def existing(d, *names, required=True):
    for n in names:
        if (d / n).exists():
            return d / n
    if required:
        sys.exit(f"ERROR: none of {names} in {d}")
    return None


def all_events_log(d):
    return existing(d, "all_events.log", "eventos_todos.log", required=False)


def evaluate(d, log, label):
    """Re-runs evaluate_test_64b.py on d/label/ with sifter.log -> log. Returns (results.md text, best S/N per file)."""
    import numpy as np
    sub = d / label
    sub.mkdir(exist_ok=True)
    for f in ("assignment.csv", "asignacion.csv", "emitter.json", "emisor.json", "server.log", "gpu.csv", "vmstat.log"):
        if (d / f).exists() and not (sub / f).exists():
            (sub / f).symlink_to(d / f)
    s = sub / "sifter.log"
    if s.exists() or s.is_symlink():
        s.unlink()
    s.symlink_to(log)
    subprocess.run([PY, str(PC / "09_evaluation" / "evaluate_test_64b.py"), str(sub)], capture_output=True, cwd=PC)
    return (existing(sub, "results.md", "resultados.md").read_text(),
            np.load(existing(sub, "best_snr_per_file.npy", "mejor_snr_por_archivo.npy")))


def start_s(r):
    return float(r.get("start_sample") or r["inicio_muestra"]) / 1000
