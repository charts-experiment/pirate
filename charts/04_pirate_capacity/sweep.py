"""Tree memory (GPU/RAM) and PCIe traffic at real time reported by `pirate_frb show dedisperser`, for ranks 13–18
and block sizes 512–8192 samples (max_gpu_clag ≈ 10 s on the GPU). It does not need compiled kernels.
Output: sweep_show.json (read by fig_capacity.py). Run it from a scratch folder: it writes one .yml per case.
"""
import re, subprocess, sys, json, itertools
from pathlib import Path
base = (Path(__file__).resolve().parent.parent / "06_configs" / "charts_64b_dm1000.yml").read_text()
res = []
for rank, nt in itertools.product([13, 14, 15, 16, 17, 18], [512, 1024, 2048, 4096, 8192]):
    clag = max(1, round(10.24 / (nt / 1000)))   # ~10 s on the GPU
    y = re.sub(r"toplevel_tree_rank: \d+", f"toplevel_tree_rank: {rank}", base)
    y = re.sub(r"time_samples_per_chunk: \d+", f"time_samples_per_chunk: {nt}", y)
    y = re.sub(r"max_gpu_clag: \d+", f"max_gpu_clag: {clag}", y)
    f = Path(f"r{rank}_nt{nt}.yml"); f.write_text(y)
    p = subprocess.run(["pirate_frb", "show", "dedisperser", str(f)], capture_output=True, text=True, timeout=300)
    o = p.stdout + p.stderr
    g = lambda k: (m := re.search(k + r":\s*([\d.]+)", o)) and float(m.group(1))
    r = dict(rank=rank, nt=nt, clag=clag, dm_max=g("dm_max"), host=g("host_zones"), gpu=g("gpu_zones"),
             h2g=g("h2g bandwidth"), max_clag=g("max_clag"), ok=p.returncode == 0)
    if not r["ok"]: r["err"] = o.strip().splitlines()[-1][:150]
    print(r, flush=True); res.append(r)
Path("sweep_show.json").write_text(json.dumps(res, indent=1))
