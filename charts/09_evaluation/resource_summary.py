#!/usr/bin/env python3
"""Per-process resource summary (file from 10_rfi_analysis/process_monitor.py): GPU memory and use, RAM and CPU
of the pirate server, the grouper, the sifter and the emitter, while the emitter is sending.

Usage: python 09_evaluation/resource_summary.py RUN_FOLDER/resources_per_process.csv
Notes: the GPU use from `nvidia-smi pmon` is a ~1 s sample every 5 s and underestimates processes that work in
bursts (pirate); the emitter's RAM includes the .fil files mapped from the disk cache.
"""
import csv
import math
import statistics as st
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
with_emitter = {r["t"] for r in rows if r["role"] == "emitter"} or {r["t"] for r in rows}
print("| Process | Max GPU memory (GiB) | Mean GPU use (%) | Max RAM (GiB) | Mean CPU (%) | CPU p90 (%) |")
print("|---|---|---|---|---|---|")
for role in ("pirate_server", "grouper", "sifter", "emitter"):
    r = [x for x in rows if x["role"] == role and x["t"] in with_emitter]
    if not r:
        continue
    cpu = sorted(float(x["cpu_pct"]) for x in r if not math.isnan(float(x["cpu_pct"])))
    print(f"| {role} | {max(float(x['gpu_mem_mib']) for x in r) / 1024:.2f} | {st.mean(float(x['gpu_sm_pct']) for x in r):.1f} | "
          f"{max(float(x['rss_gib']) for x in r):.1f} | {st.mean(cpu):.0f} | {cpu[int(0.9 * len(cpu))]:.0f} |")
