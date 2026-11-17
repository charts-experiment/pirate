"""Memory and real-time capacity of pirate for 64 beams × 21 504 channels, sweeping rank, block size, max_gpu_clag
and number of trees. For each case: `pirate_frb show dedisperser` (tree state in GPU/RAM, PCIe at real time) and
`pirate_frb time_dedisperser -H -n 24` (reserved memory, real-time beams with the GPU at 100 %).
Output: timings.json (read by fig_capacity.py). Run it from a scratch folder: it writes one .yml per case.
"""
import re, subprocess, json, itertools, statistics
from pathlib import Path
base = (Path(__file__).resolve().parent.parent / "06_configs" / "charts_64b_dm1000.yml").read_text()
T1 = "  - {num_early_triggers: 0, max_width: 32, wt_dm_downsampling: 64, wt_time_downsampling: 64}\n"
T2 = "  - {num_early_triggers: 0, max_width: 16, wt_dm_downsampling: 64, wt_time_downsampling: 64}\n"
T3 = "  - {num_early_triggers: 0, max_width:  8, wt_dm_downsampling: 64, wt_time_downsampling: 64}\n"
def cfg(rank, nt, clag, ntrees=1):
    y = re.sub(r"toplevel_tree_rank: \d+", f"toplevel_tree_rank: {rank}", base)
    y = re.sub(r"time_samples_per_chunk: \d+", f"time_samples_per_chunk: {nt}", y)
    y = re.sub(r"max_gpu_clag: \d+", f"max_gpu_clag: {clag}", y)
    y = y.replace(T1, T1 + (T2 if ntrees > 1 else "") + (T3 if ntrees > 2 else ""))
    return y
cases = []
for rank, nt in itertools.product([13, 14, 15, 16], [512, 1024, 2048, 4096, 8192]):
    cases.append(("dm_nt", rank, nt, max(1, round(10240 / nt)), 1))
for ntr in (2, 3):
    for clag in (5, 2, 1):
        cases.append(("trees", 15, 2048, clag, ntr))
for clag in (1, 2, 3, 5, 8, 12, 16):
    cases.append(("clag", 15, 2048, clag, 1))
cases.append(("trees", 16, 2048, 1, 2))
res = []
for kind, rank, nt, clag, ntr in cases:
    f = Path(f"t_{kind}_r{rank}_nt{nt}_g{clag}_t{ntr}.yml"); f.write_text(cfg(rank, nt, clag, ntr))
    r = dict(kind=kind, rank=rank, nt=nt, clag=clag, ntrees=ntr)
    s = subprocess.run(["pirate_frb", "show", "dedisperser", str(f)], capture_output=True, text=True).stdout
    for k, n in (("host_zones", "ring_host"), ("gpu_zones", "ring_gpu"), ("h2g bandwidth", "h2g_rt")):
        m = re.search(k + r":\s*([\d.]+)", s); r[n] = m and float(m.group(1))
    r["dm_max"] = max(float(x) for x in re.findall(r"dm_max:\s*([\d.]+)", s)) if "dm_max" in s else None
    try:
        p = subprocess.run(["pirate_frb", "time_dedisperser", "-H", "-n", "24", str(f)], capture_output=True, text=True, timeout=900)
        o = p.stdout + p.stderr
        m = re.search(r"gpu=([\d.]+) GB sync, cpu=([\d.]+) GB", o)
        if m: r["gpu_GB"], r["cpu_GB"] = float(m.group(1)), float(m.group(2))
        it = re.findall(r"real-time beams = ([\d.]+).*h2g_bw = ([\d.]+)", o)[-8:]
        if it:
            r["beams_rt"] = statistics.median(float(a) for a, _ in it)
            r["h2g_full"] = statistics.median(float(b) for _, b in it)
        else:
            r["err"] = o.strip().splitlines()[-1][:160]
    except subprocess.TimeoutExpired:
        r["err"] = "timeout"
    print(r, flush=True); res.append(r)
    Path("timings.json").write_text(json.dumps(res, indent=1))
