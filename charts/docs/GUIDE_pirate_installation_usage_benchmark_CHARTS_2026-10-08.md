# pirate (pirate_frb): installation, usage and first benchmark for CHARTS

**Date:** 2026-10-08 · **Machine:** RTX 4090 (sm_89), CUDA 13.1, gcc 13.3, 314 GiB RAM
**Upstream docs:** https://kmsmith137.github.io/pirate/ (TeX notes: `dedispersion.pdf`, `detrending.pdf`, `variance_map.pdf`, `inpainting.pdf`)

pirate is the real-time GPU FRB search that Kendrick Smith is writing for CHORD.
It is a **tree** dedisperser (bonsai-like, but on the GPU) in C++/CUDA with a Python interface (`pirate_frb`).
It includes a server that receives data from the X-engine over TCP, a ring buffer, data writing over RPC (gRPC) and peak-finding.

This document is the working log of the first days with pirate. Paths like `projects/pirate_charts/...` refer to the
working folder used at the time; the maintained code is now in `charts/` (see `docs/STRUCTURE.md`).

---

## 1. What was installed and where

| Item | Path |
|---|---|
| Conda environment (Python 3.13) | `<pirate environment>` |
| ksgpu source (dependency) | `<sources>/ksgpu` (pip editable) |
| pirate source | `<sources>/pirate` (pip editable, v1.5.0) |
| CHARTS configs and logs | `<working folder>` |

Activating the environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <pirate environment>
pirate_frb --help
```

**Check:** `pirate_frb test -n 1` passes completely (~1 min), with no failures.

### Problems found during installation (for reinstalling)

1. **`pip` points to a Python 2.7 pip** (`~/.local/bin/pip` wins in the PATH). Inside the environment always use `python -m pip ...`.
2. **`clang-format` is missing.** The kernel generator needs it, but the `environment_*.yml` files do not declare it. It was installed with `conda install -c conda-forge clang-format`.
3. **Race in `make -j`.** `asdf-cxx/include/asdf/config.hxx` is not generated in time. Running `make asdf-cxx/include/asdf/config.hxx` before `make -j 32` is enough.
4. **Architecture.** It was built only for sm_89 with a `config.mk` (gitignored) in each repo: `NVCC_ARCH = -gencode arch=compute_89,code=sm_89`. pirate takes ~2 min with 32 threads.

To update: `git pull && make -j 32` in `ksgpu` and then in `pirate`, with the environment active. Since it is an editable install, the pip step does not need to be repeated.

---

## 2. Key concepts

- **Tree dedispersion:** its cost is ~O(N_f · N_t · log N_DM), not O(N_f · N_t · N_DM) like brute force (AstroAccelerate, dedisp). That is why the throughput is much higher. In exchange, the DM grid and the delays are approximate; the details are in `dedispersion.pdf`.
- **`toplevel_tree_rank` (R):** sets the number of DMs, 2^R, and the maximum delay, 2^R samples. With 1 ms and 300–501.6 MHz:
  - R = 15 → DM 0–1107
  - R = 16 → DM 0–2213
- **`primary_trees`:** additional trees that cover higher DMs. Each one is applied to data downsampled ×2 in time.
- **`frequency_subband_counts`:** each tree also searches in subbands, which gives sensitivity to narrow-band FRBs. They are computed with `pirate_frb dev make_subbands FMIN FMAX 0.1 -r 4`.
- **Peak-finding:** done in the same kernel (boxcars up to `max_width`). The output is a coarse (DM, time) matrix with the **maximum S/N** (`out_max`) and a token `out_argmax` that encodes fine DM, time, width and subband.
  - The S/N is normalized with analytic weights from the per-channel variances.
- **Input format:** **int4 with float16 scale/offset** per (beam, channel, block of 256 samples). The value −8 means a masked sample.
  - It arrives over TCP (protocol in `notes/network_protocol.md`) or, in offline mode, from ASDF files (`frame_b{BEAM}_t{CHUNK}.asdf`).
- **Pre-generated kernels:** kernels exist only for fixed combinations of (dtype, dd_rank, max_width, subbands, early triggers). They are listed in `makefile_helper.py`, under `production_kernels`.
  - A new config with other subbands requires adding rows there and rebuilding.
  - In contrast, the number of channels, the band, `time_sample_ms` and the number of beams can be changed without rebuilding.

---

## 3. Main commands

```bash
# Inspect a config: dedispersion plan, DM range per tree and memory
pirate_frb show dedisperser -c -t CONFIG.yml

# GPU benchmark: H2G upload of int4, dequantization, dedispersion and peak-finding
pirate_frb time_dedisperser -n 30 -b NBEAMS -H -g CLAG CONFIG.yml
#   -b  number of beams actually allocated in memory
#   -H  without hugepages (needed here: no hugepages reserved)
#   -g  max_gpu_clag: less ring buffer in VRAM and more in RAM

# Unit tests
pirate_frb test -n 1

# Full "toy" search over loopback, 4 terminals in this order
# (exact commands in notes/quick_start.md):
pirate_frb run toy_sifter 127.0.0.1:7500
pirate_frb run toy_grouper ...
pirate_frb run server configs/frb_server/toy.yml configs/dedispersion/toy.yml
pirate_frb run fake_xengine -f -g 30 ...      # injects simulated FRBs every 30 s
pirate_frb rpc status | start_stream | show_streams | rand_write ...

# Offline dedispersion of saved data (1 beam; provisional peak-finding: max S/N per chunk)
pirate_frb run offline_dedisperser ACQDIR CONFIG.yml
```

From Python, `pirate_frb.OfflineDedisperser(config)` processes one `AssembledFrame` per chunk. Inside `with od.dedisperse(frame) as out:` one reads `out.out_max[itree]`, with shape (beams, coarse_ndm, coarse_nt).

---

## 4. First benchmark with CHARTS parameters

**Config:** `projects/pirate_charts/configs/charts_sb0_r15.yml`

- 300–501.6 MHz, **21 504 channels**, 1 ms, fp16
- Rank 15: one tree, **DM 0–1107**, boxcar ≤ 32 ms
- No subbands: reuses the kernels already compiled (`chord_sb0` style)

"Capacity" is the number of beams the GPU would process in real time at the measured rate (`real-time beams`). If the allocated beams are fewer than the capacity, the configuration runs in real time.

| Allocated beams | clag | VRAM | Pinned RAM | Capacity (r.t. beams) | Load | Real time? |
|---:|---:|---:|---:|---:|---:|:---:|
| 16 | 5 | — | — | 765 | 2 % | ✅ |
| 64 | 5 | 15.6 GB | 21.5 GB | 761 | 8 % | ✅ |
| 96 | 2 | 9.5 GB | 45 GB | 481 | 20 % | ✅ |
| 128 | 1 | 7.6 GB | 65 GB | 396 | 32 % | ✅ |
| 192 | 2 | 17.2 GB | 91 GB | 481 | 40 % | ✅ |
| **256** | **1** | **13.4 GB** | **130 GB** | **405** | **63 %** | ✅ |
| 320 | 1 | 16.2 GB | 162 GB | 392 | 82 % | ✅ (little margin) |
| 384 | 0 | 10.0 GB | 204 GB | 335 | 115 % | ❌ |

Reading:

- **Compute is not the limit.** If everything fits in VRAM, the GPU handles ~760 beams.
- **The limit is memory.** The dedispersion ring buffer needs ~0.24 GB of VRAM plus ~0.34 GB of RAM per beam with clag 5.
  - Lowering `max_gpu_clag` moves the ring buffer to RAM and the bottleneck becomes PCIe: H2G ≈ 22.5 GB/s, practically saturated on PCIe 4.0 x16.
- **Practical ceiling:** ~250–300 beams of 21 504 channels per 4090 (`-g 1`, ~130–160 GB of RAM), with a 20–40 % margin.
  - For reference, the estimate for s-engine/AstroAccelerate in the plan of 2026-09-25 was ~70–100 beams.
- With clag 2 the capacity rises to ~480 beams, but the VRAM is enough for only ~200 beams.

**Note: this measures only the GPU part.** It does not include:

- network reception or frame assembly (CPU);
- **RFI masking**, which is not yet implemented in pirate (it is the author's next priority);
- grouper and sifter, which at that time were "toy" versions;
- the 8-bit to int4 conversion.

---

## 5. Differences with CHARTS and work needed (as of 2026-10-08)

1. **Quantization to int4.** CHARTS delivers 8 bits and pirate only accepts int4 with scale/offset every 256 samples.
   - The data must be requantized and the S/N loss measured on the comparison set. It was expected to be small but was not yet measured (later measured: see `docs/ANALYSIS_pirate_quantization_and_group3_21504_2026-10-09.md`).
2. **No RFI.** With real data the S/N is not reliable until masking exists.
   - In the meantime, samples can be flagged as −8 (masked) with the same masking used in the other pipelines.
3. **CHARTS subbands.** `make_subbands 300 501.6 0.1 -r 4` gives counts `(0,1,6,3,1)`, a combination without compiled kernels.
   - Using it requires adding 4 rows `('fp16', 8|7, W, 1, (0,1,6,3,1), et)` to `makefile_helper.py` and rebuilding. It may cost throughput.
4. **Data input.**
   - **Offline**, to evaluate recall and FP with our `.fil` files: write a `.fil → ASDF frames` converter or feed `OfflineDedisperser` from Python.
     - Offline peak-finding is provisional. Candidates (DM, t, width, S/N) must be extracted from `out_max`/`out_argmax` and grouped by us.
   - **Real time:** implement an emitter of the X→FRB v2 protocol (TCP, minichunks of 256 samples) from Kotekan/CHARTS. The server configures itself from the metadata it receives.
5. **Unstable API.** The project is "experimental" and in active development (1.5.0), so formats and protocol may change.

---

## 6. Offline search on .fil files and comparison with the other pipelines (2026-10-08, second part)

### Code

- `pirate_fil_search.py` (now `03_offline_search/`): searches for FRBs with pirate in an 8-bit `.fil`.
  - **It does not go through int4 or ASDF.** It normalizes per channel and writes float16 directly into the input buffer of the `GpuDedisperser` (`get_input()`).
  - It decodes `out_argmax` with `plan.decode_argmax/decode_argmax2` to get DM, t, width and subband.
  - Grouping: 3×3 local maxima + non-maximum suppression (0.1 s and 10 % in DM).
  - `--int4 N` simulates the requantization of the network protocol, with N levels per σ.
  - The frequency order is verified: pirate's channel 0 is the **lowest** frequency, so the `.fil` (foff < 0) is reversed.
- `charts_search.yml` (now `06_configs/`): rank 16 with two trees. Covers DM 0–~2200 with boxcar ≤ 32 samples and DM ~2200–4400 with a 32 ms boxcar. No subbands.
- pirate was added as the fifth pipeline of the offline benchmark.
- `calibrate_snr.py` (now `03_offline_search/`): injects boxcar pulses of known S/N into gaussian noise.

### Calibration with injected pulses (true S/N 20, no intra-channel smearing)

| Channels | W = 1 ms | W = 4 ms | W = 16 ms |
|---|---|---|---|
| 21 504 | 0.63–0.72 | 0.86–0.91 | 0.92–1.09 |
| 672 | 0.32–0.42 (no detection at DM ≥ 1000) | 0.31–0.79 | 0.60–0.90 |

- pirate assumes channels that are narrow compared with the delay step between channels, as in CHORD or CHARTS with 21 504 channels. With 672 channels and narrow pulses it loses S/N.
- DM error < 0.1 % and time error ≤ 1.4 ms with 21 504 channels.
- **int4 versus float16** (same noise realization): loss of 0–1 % with 2 levels/σ and ~2 % with 3 levels/σ. Negligible.

### Comparison: 300 files of the group-2 comparison set (672 channels, 100 A + 100 B + 100 NEG), boxcar ≤ 32 ms

The four classical pipelines come from the bc32 campaign `20260925_042706`, on the same files.

With S/N ≥ 5:

| | Recall A | Recall B | Noise FP/h | Non-FRB FP (% NEG) |
|---|---|---|---|---|
| Heimdall | 87 | 58 | 444 | 45 |
| AA | 100 | 85 | 979 | 95 |
| CASM-Hella | 95 | 63 | 865 | 66 |
| PRESTO | 99 | 66 | 3233 | 63 |
| **pirate** | **100** | **79** | 2999 | 88 |

At equal noise FP rate, **10 events/h**:

| | threshold | Recall A | Recall B | Non-FRB FP (% NEG) |
|---|---|---|---|---|
| Heimdall | 5.80 | 73 | 54 | 32 |
| AA | 6.14 | 98 | 63 | 56 |
| CASM-Hella | 6.80 | 49 | 43 | 27 |
| PRESTO | 6.12 | 77 | 54 | 35 |
| **pirate** | 6.16 | **100** | **68** | 67 |

- At equal noise FP rate, pirate has the best recall in A and in B.
- Its non-FRB FP is high, like AA's: it also detects the negatives, which are dispersed signals with a non-physical DM index.
- With 100 files per set, each percentage has an error of ±4–5 %.
- Search time: **0.76 s per 119 s file** (672 channels, 1 beam), counting only the search.

---

## 7. Real-time test with ring buffer and maximum number of beams (2026-10-08, third part)

### Chain tested (all on this machine, over loopback)

`fake_xengine` (TCP, int4) → `FrbServer` (reception → frame ring buffer in RAM → H2G → dedispersion + peak-finding on the GPU) → `toy_grouper` (threshold 10) → `toy_sifter`.

- The fake X-engine stays at most 4 chunks ahead of the server. That is why the rate `beams=X/N` printed by the server is the real capacity of the full chain.
- The frame ring buffer takes all the RAM left over in `host_memory_per_server` after the dedispersion.

### Dedispersion config chosen: `charts_rt_r15t3`

Rank 15 with 3 trees, no subbands:

| Tree | Resolution | DM |
|---|---|---|
| 0 | 1 ms | 0–1107 |
| 1 | 2 ms | 1107–2213 |
| 2 | 4 ms | 2213–4427 |

Maximum boxcar of 32 ms in the three trees.

Rank 16 (a single 1 ms tree up to DM ~2200) costs ~3.7 GB of RAM per beam and is enough for only ~64 beams. Rank 15 with 3 trees costs ~1.3 GB per beam.

`time_dedisperser` (GPU only, 21 504 channels):

| Beams | clag | VRAM | RAM | Capacity (real-time beams) |
|---:|---:|---:|---:|---:|
| 128 | 1 | 11.3 GB | 167 GB | 260 |
| 160 | 2 | 19.6 GB | 202 GB | 298 |
| 192 | 1 | 15.7 GB | 250 GB | 264 |
| 128 | 3 | 22.2 GB | 156 GB | 341 |

### Results of the full chain

| Run | Beams | Receivers | Data | Frame ring buffer | Capacity (equiv. beams) | Real time | GPU | Idle CPU |
|---|---:|---:|---|---|---:|---|---:|---:|
| b128_junk | 128 | 1 | zeros | ~31 chunks = ~64 s | 151.5 | 1.18× | 52 % | 87 % |
| b160_2rx_junk | 160 | 2 | zeros | ~12 chunks = ~25 s | 222 | **1.39×** | 89 % | 81 % |
| b160_2rx_frb | 160 | 2 | gaussian + FRBs S/N 20 | ~25 s | 200 | **1.25×** | 87 % | 43 % |

- **With one receiver**, the TCP input is limited to ~13.5 Gbps. With two receivers (`data_ip_addrs: [[':5000', ':5001']]`) it reaches ~19.7 Gbps and the limit becomes the GPU.
- **Injected FRBs with 160 beams:** 1557 of 1557 evaluable ones were detected (100 %) in the three trees and at all widths. Measured S/N: median 18.8 for 20 injected (p10 16.2). With threshold 10 there were no false events.
- **The maximum for this machine is ~160 beams, set by the RAM.**
  - With 160 beams ~2–4 GB of RAM are left, not even enough to start a new Python process (that is why `rpc status` fails with an MKL error).
  - More beams shrink the ring buffer below ~25 s. The full sweep at DM 1000 in 300–501.6 MHz lasts ~30 s, so a dump per trigger needs at least that.
  - Towards 176 beams the GPU would also be at ~98 %.

### Caveats

- The fake X-engine runs on the same machine. With gaussian data it uses ~45 % of the CPU and takes some margin from the server (1.25× instead of 1.39×).
- The injected FRBs come from pirate's simulator: ideal pulses, without RFI or intra-channel smearing. The grouper is the "toy" one (one event per chunk and beam above the threshold).
- Not tested:
  - ring-buffer dumps over RPC (`write_files`, `start_stream`);
  - a real network card, since the test was over loopback;
  - long runs (hours).
