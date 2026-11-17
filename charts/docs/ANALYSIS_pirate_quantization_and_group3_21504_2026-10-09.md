# pirate at CHARTS, the 21 504-channel set (group 3) and 8 → 4-bit quantization

**Dates:** 2026-10-08 and 2026-10-09
**Machine:** RTX 4090 (24 GB), Xeon w5-3435X (32 threads), 314 GiB of RAM, NVMe
**Related document:** `GUIDE_pirate_installation_usage_benchmark_CHARTS_2026-10-08.md` (pirate installation and commands)

This is the working log of those days. The code mentioned has since moved into `charts/` (see the table in section 9).

---

## 0. Summary

| Topic | Result |
|---|---|
| **pirate installed** | conda environment + `ksgpu` and `pirate` sources. The unit tests pass. |
| **Capacity, GPU only** | 21 504 channels, 1 ms, DM ≤ 1107: ~760 beams in real time if everything fits in VRAM; ~250–300 with the ring buffer in RAM. The limit is memory + PCIe, not compute. |
| **Offline search** (672 channels, 300 files) | At equal noise FP rate (10/h), pirate has the best recall: A 100 %, B 68 %. Its non-FRB FP is high (67 %), like AA's. |
| **Real time with ring buffer** (full chain) | Maximum for this machine: **160 beams** of 21 504 channels, DM ≤ 4427; 1.25–1.39× faster than real time. **100 % of the 1557 injected FRBs detected.** The limit is the RAM. |
| **RFI** | The `lineas10` model (0 dB) prevents any detection (P8). The 64-beam test is done **without RFI**; new RFI measurements are coming. |
| **Group 3** (21 504 channels) | Generator `12_generate_comparison_set_21504.py` written and validated with 40 files. Full generation **running** since 2026-10-09. |
| **8 → 4-bit quantization** | With **n = 2 levels/σ and centre 32.5**, 98.97 % of the S/N of weak bursts is kept, with a cap of ±3.5σ (46 counts). Bursts exceeding ~14 counts per pixel above the floor are clipped: they are still detected, but with an underestimated S/N. |

---

## 1. pirate: what it is and how it was installed

pirate (`pirate_frb`, K. Smith, CHORD) is a real-time FRB search for the GPU. It uses **tree dedispersion**, with a cost ~O(N_f·N_t·log N_DM), instead of brute force. It has a server that receives int4 over TCP from the X-engine, a ring buffer in RAM, peak-finding on the GPU and a grouper/sifter connected over gRPC.

- **Installation problems solved:**
  - `pip` points to a Python 2.7: use `python -m pip`.
  - `clang-format` was missing: installed in the environment.
  - Race in `make -j` with `asdf/config.hxx`.
  - `config.mk` fixes the build to sm_89.
- **Frequency order:** pirate's channel 0 is the **lowest** frequency. The `.fil` (foff < 0) has to be reversed.

---

## 2. GPU capacity (`pirate_frb time_dedisperser`)

Measures H2G upload of int4 + dequantization + dedispersion + peak-finding. Does not include network, RFI or grouper.

### DM ≤ 1107 (rank 15, one tree, no subbands)

| Allocated beams | max_gpu_clag | VRAM | RAM | Capacity (real-time beams) |
|---:|---:|---:|---:|---:|
| 64 | 5 | 15.6 GB | 21.5 GB | 761 |
| 128 | 1 | 7.6 GB | 65 GB | 396 |
| 256 | 1 | 13.4 GB | 130 GB | 405 |
| 320 | 1 | 16.2 GB | 162 GB | 392 |
| 384 | 0 | 10.0 GB | 204 GB | 335 ❌ |

### DM ≤ 4427 (rank 15, three trees: DM 0–1107 at 1 ms, 1107–2213 at 2 ms, 2213–4427 at 4 ms)

| Beams | clag | VRAM | RAM | Capacity |
|---:|---:|---:|---:|---:|
| 64 | 1 | 6.9 GB | 84 GB | 255 |
| 128 | 2 | 16.2 GB | 162 GB | 299 |
| 192 | 1 | 15.7 GB | 250 GB | 264 |

Rank 16 (a single 1 ms tree up to DM ~2200) costs ~3.7 GB of RAM per beam, which gives a maximum of ~64 beams. **The limit is the RAM and PCIe (~22 GB/s), not compute.**

---

## 3. Offline search on `.fil` files and comparison with the other pipelines

- `pirate_fil_search.py` (now `03_offline_search/`) reads the 8-bit `.fil`, normalizes per channel and writes float16 **directly** into the GPU buffer. It does not go through int4 or ASDF.
- It decodes the candidates with `decode_argmax`, groups with local maxima + non-maximum suppression, and returns `(dm, t_top, snr, width)`.
- Integrated as the 5th pipeline of the offline benchmark.

**300 group-2 files** (672 channels: 100 A, 100 B, 100 NEG), boxcar ≤ 32 ms:

| | Recall A (S/N ≥ 5) | Recall B | Noise FP/h | Non-FRB FP | **Recall A at 10 FP/h** | **Recall B at 10 FP/h** | Non-FRB FP at 10 FP/h |
|---|---|---|---|---|---|---|---|
| Heimdall | 87 | 58 | 444 | 45 % | 73 | 54 | 32 % |
| AA | 100 | 85 | 979 | 95 % | 98 | 63 | 56 % |
| CASM-Hella | 95 | 63 | 865 | 66 % | 49 | 43 | 27 % |
| PRESTO | 99 | 66 | 3233 | 63 % | 77 | 54 | 35 % |
| **pirate** | **100** | 79 | 2999 | 88 % | **100** | **68** | 67 % |

- Each percentage has an error of ±4–5 %, since there are 100 files per set.
- pirate's search: 0.76 s per 119 s file, dominated by Python overhead.
- At S/N ≥ 5, pirate produces many noise FPs because it tries 65 536 DMs. The practical threshold is ~6–7.

### S/N calibration (injected boxcar pulses, true S/N 20)

| Channels | W = 1 ms | W = 4 ms | W = 16 ms |
|---|---|---|---|
| 21 504 | 63–72 % | 86–91 % | 92–109 % |
| 672 | 32–42 % (no detection at DM ≥ 1000) | 31–79 % | 60–90 % |

pirate assumes channels that are narrow compared with the delay between channels. At 21 504 channels, which is the CHARTS case, it works well. DM and time come out with error < 0.1 % and ≤ 1.4 ms.

---

## 4. Real time with ring buffer (full chain on this machine)

`fake_xengine` (TCP, int4) → `FrbServer` (reception → frame ring buffer in RAM → GPU) → `toy_grouper` → `toy_sifter`. All over loopback.

| Run | Beams | Receivers | Data | Ring buffer | Speed vs. real time | GPU |
|---|---:|---:|---|---|---|---:|
| b128_junk | 128 | 1 | zeros | ~64 s | 1.18× | 52 % |
| b160_2rx_junk | 160 | 2 | zeros | ~25 s | **1.39×** | 89 % |
| b160_2rx_frb | 160 | 2 | gaussian + FRBs of S/N 20 | ~25 s | **1.25×** | 87 % |

- **160 beams with injected FRBs:** 1557 of 1557 detected (100 %), in the 3 trees and at all widths. Measured S/N: median 18.8 for 20. With threshold 10, no false events.
- **With one TCP receiver** the input stalls at 13.5 Gbps. With two it reaches 19.7 Gbps.
- **The limit is the RAM:** with 160 beams 2–4 GB are left, and a new Python process cannot even start (`rpc status` fails with an MKL error).
- **Operational lessons:**
  - Do not edit a `.sh` while it runs: bash rereads it.
  - `pgrep -f` finds itself in the wrapper's command line; the pattern has to be anchored to the environment's Python.

---

## 5. RFI: decision

- **The RFI of the earlier tests is `lineas10`:** model v2 (line threshold 10 dB), 0 dB, k = 5, impulses at 1/s, seed 42.
- **P8 (diagnosis of 2026-10-01):** with that model, no A burst is detected (0 of 5) and in B only those with S/N ≳ 80.
  - The cause is the fast RFI measured with **one antenna at 100 ms**, applied to the beam at 0 dB and held for 100 samples in all channels. The noise of the dedispersed series rises ×7–14.
  - Attenuation sweep: S/N 0.7 (0 dB), 1.6 (10 dB), 1.9 (20 dB) and 3.0 (30 dB), against 7.3 without RFI.
- **Decision of the CHARTS team (2026-10-09):** the 64-beam test is done **without RFI**. New measurements are coming to model the RFI more realistically. (They arrived later: the site baseband of 2026-09-02, see `05_site_rfi/` and `docs/RESULTS_64beams.md`.)

---

## 6. Group 3: comparison set at 21 504 channels

### Definition

Implemented in `12_generate_comparison_set_21504.py` (now `01_simulator/`).

| Parameter | Value |
|---|---|
| Draws | **the same 1000 + 1000 as group 2** (`group2_drawn_parameters.csv`), seed 20260923 |
| Physics | `model()` of `07_`: intra-channel smearing, spectrum normalized to the brightest channel, energy-conserving scattering, ×4 oversampling |
| Grid | **21 504 channels** = each of the 672 channels (300.45 kHz) split into 32 (**9.389 kHz**); centres from 299.855 to 501.745 MHz |
| Noise | σ = SEFD / √(Δν·τ) = **768.5 Jy** per channel and sample (135.9 Jy at 672 channels) |
| A | S/N 8, **recomputed** with the 21 504-channel model |
| B | **S0 = 616.87 Jy**, the same as in group 2 |
| Negatives | pair of A_i (its `dm_index_negativo`, scale of A_i) |
| Partial | full burst at S/N 15 (recomputed), same edge and target fraction |
| Arrival | t_top (at 501.6 MHz) drawn between 3 and 5 s, with a seed per id |
| Length | **variable**: ⌈t_top + sweep + pulse tail + 1 s⌉; in the pilot test, 10–77 s |
| Format | **8-bit** `.fil`: floor 32, **1σ = 4 counts**, same quantization as group 2 |
| Output | `manifest.csv` per set, `config.json`, `_done.jsonl` (allows resuming) |

### Validation (pilot test of 40 files)

Script: `validate_21504.py` (now `02_quantization/`).

- **Noise:** mean 32.000 and σ 4.010 counts, exactly as designed. The noise of the dedispersed series comes out at 1.00 ± 0.02.
- **Set A:** measured − expected S/N = **+0.19 ± 0.76**, as expected with a single noise realization per file.
- **B_001** (DM 727, same FRB as A_001): the sweep follows the theoretical curve and the measured S/N is 154.9 against 153.4 expected.
- **Gain of the fine grid:** A_000 (DM 1605) needs 86 Jy for S/N 8, against 287 Jy at 672 channels (×3.3). It agrees with the population study.
- **Reproducibility:** the float reconstruction with the same seed reproduces the `.fil` files **byte for byte**.

### Full generation

- Launched on 2026-10-09 with 24 processes. ETA ~2.5–3 h, ~4.4 TB.
- If it stops, relaunching the same command resumes where it left off:
  ```
  cd 01_simulator
  python 12_generate_comparison_set_21504.py --workers 24 --out /path/to/group3
  ```

---

## 7. Quantization: from the 8 bits of the set to pirate's 4 bits

### 7.1 Concepts

- **σ** is the noise standard deviation of **one channel in one 1 ms sample**.
  - **Physically** it comes from the radiometer equation: 768.5 Jy at 21 504 channels.
  - **In the `.fil`** the scale was set so that the floor is at count 32 and **1σ = 4 counts**.
  - **In operation** it would be estimated per channel with a running window of several seconds (standard deviation or 1.4826·MAD). The window must be much longer than a pulse, so as not to subtract it.
- **Why 4 bits:** pirate receives int4 with float16 scale and offset. It is a requirement of its network protocol, not of the instrument. CHARTS delivers intensity in 8 bits.
- **Why there are negative values:** the 4-bit code (−7…+7) stores the **fluctuation around the mean**. The offset stores the floor and the intensity is reconstructed as offset + scale·q, which is still positive. −8 is reserved for masked samples.
- **"n levels per σ"** = how many 4-bit codes 1σ spans:
  - step Δ = σ/n
  - q = round((x − μ)/Δ), clipped to ±7
  - cap = 7σ/n
  - Examples: n = 1 → ±7σ; **n = 2 → ±3.5σ**; n = 3 → ±2.33σ.

### 7.2 Theory (weak signal in gaussian noise)

Fraction of S/N kept by a quantizer Q, with tₖ the thresholds, Jₖ the jump of Q at each threshold and φ the gaussian density:

  η = Σₖ Jₖ·φ(tₖ) / √Var[Q(x)]

Script: `02_quantization/quantization_theory.py`.

**From continuous data:**

| n | Step | Cap | S/N kept | Noise samples clipped |
|---|---|---|---|---|
| 1 | 1σ | ±7σ | 96.1 % | 0 |
| 1.5 | 0.67σ | ±4.7σ | 98.2 % | 0 |
| 2 | 0.5σ | ±3.5σ | 99.0 % | 0.02 % |
| 2.83 | 0.35σ | ±2.5σ | **99.35 % (optimum)** | 0.8 % |
| 3 | 0.33σ | ±2.3σ | 99.3 % | 1.2 % |
| 4 | 0.25σ | ±1.75σ | 98.7 % | 6 % |

**From the 8 bits of the set** (the real chain, `.fil` → emitter):

| Conversion | S/N kept |
|---|---|
| 8 bits (the `.fil`) | 99.74 % |
| n = 2 with centre 32 (`q = rint((x−32)/2)`) | 98.22 % (ties at .5 create unequal intervals) |
| **n = 2 with centre 32.5** (`q = rint((x−32.5)/2)`) | **98.97 %**; each code groups exactly 2 counts |
| n = 3 (`q = rint((x−32)·3/4)`) | 99.03 % |
| n = 1 | 95.35 % |

### 7.3 Measurement with the same noise and the same signal

Script: `02_quantization/quantization_21504.py`. It reconstructs the exact float of each file of the pilot test and quantizes it in every way.

| Quantization | Theory | Measured on the bright B (S/N 57–156) |
|---|---|---|
| 8 bits | −0.26 % | −0.2 to −0.4 % |
| 4 bits, 3 levels/σ | −1.0 % | −1.0 to −1.6 % |
| 4 bits, 2 levels/σ (centre 32) | −1.8 % | −2.2 to −2.8 % |
| 4 bits, 1 level/σ | −4.7 % | −4.9 to −5.8 % |
| pirate-style 4 bits (scale/offset per block of 256 samples) | −1.8 % + ~width/256 | −1.8 to −6 %; **A_007 (scattering 26 ms): −22 %** |

- For S/N 8 bursts the points scatter by ±5 %: the quantization error adds some new noise. The bright bursts measure the clean loss.
- **pirate's per-block offset is an implementation issue:** the block mean includes the pulse and subtracts it. A long-term offset must be used.

### 7.4 Range: what happens with strong bursts

Script: `02_quantization/fig_quantization_range.py`. It uses the same B_001 pulse scaled, with the same noise.

| Scale | Peak per pixel | Counts above the floor | Real S/N | 8 bits | 4b n = 2 | 4b n = 3 | 4b n = 1 |
|---|---|---|---|---|---|---|---|
| ×1 | 0.8σ | 3 | 153 | 100 % | 99 % | 98 % | 96 % |
| ×3 | 2.4σ | 10 | 460 | 100 % | 98 % | 92 % | 96 % |
| ×5 | 4σ | 16 | 767 | 100 % | 93 % | 79 % | 96 % |
| ×10 | 8σ | 32 | 1534 | 100 % | 70 % | 53 % | 94 % |
| ×30 | 24σ | 96 | 4603 | 100 % | 32 % | 23 % | 53 % |
| ×100 | 80σ | 320 (saturates at 255) | 15 343 | 92 % | 12 % | 9 % | 21 % |

(The n = 2 column was measured with centre-32 rounding; in these cases clipping dominates, so centre 32.5 barely changes it.)

- **The S/N is that of the sum.** A B_001 pixel is barely ~3 counts above the floor. The S/N of 153 comes from summing ~6900 channels × ~28 ms ≈ 190 000 pixels: S/N ≈ (mean height ~0.35σ) × √190 000 ≈ 150.
- **Clipping acts on each pixel before summing.** With n = 2, anything above **46 counts** (+3.5σ) is stored as 7.
- **When a burst is "strong":** peak per pixel ≈ S/N / √(channels with emission × width in ms).
  - Broadband and ~10 ms (like B_001): saturates at S/N ≳ 400.
  - 1 ms and narrow band (~2000 channels): saturates at S/N ≳ 90.
  - In the B bursts of the pilot test, the peak per pixel ranges from 0.01 to 0.8σ.
- **Consequence:** they are still detected, but the S/N, fluence and morphology come out underestimated. pirate's ring-buffer dumps are also in 4 bits. Assuming the standard flux distribution (N ∝ S^−1.5), this would be ~1 % of the events with S/N > 100 and ~0.2 % with S/N > 300.
- **Mitigations:**
  1. Keep or dump the 8 bits on the CHARTS side: detect with 4 bits and measure with 8.
  2. Flag events with many samples at ±7: their S/N is a lower bound.
  3. Try a variable scale per block, with a long-term offset.

### 7.5 Recommendation for the emitter to pirate

```
q = round((x − 32.5) / 2), clipped to ±7     scale = 2 counts (0.5σ), offset = 32.5
```

- n = 2 with centre 32.5 keeps 98.97 % of the S/N of weak bursts, practically the same as n = 3 (99.03 %).
- It has **50 % more range** than n = 3: ±3.5σ, i.e. 46 counts.
- Each code corresponds exactly to 2 counts of the `.fil`.
- The set stays in 8 bits: it is what CHARTS delivers, it serves all the pipelines, and the conversion is an emitter step that can be changed.

This is the `fixed` quantization of `07_emitter/quantization.py`.

### 7.6 Figures

The scripts of `02_quantization/` write their figures to `02_quantization/plots_21504/`:

| Figure | Content |
|---|---|
| quantization explained | noise histograms, dedispersed pulse and theory/measurement bars |
| quantized pulses B_001 | the same pulse with each quantization: sweep, aligned at the DM and raw pixels |
| per-file theory | S/N loss file by file, with the theoretical line |
| quantization range B_001 | range loss: S/N kept versus brightness and the ×30 pulse flattened |
| quantization in counts | everything in counts: 8-bit noise, count → code conversion, bright pulse in 8 bits and in 4 bits |

---

## 8. Next steps (as of 2026-10-09)

1. **Finish group 3:** generation running (section 6).
2. **64-beam emitter for pirate:** 8 → 4 bits with n = 2 and centre 32.5; exact 1.000 ms clock (`dt_ns_per_seq = 5000`, `seq_per_frb_time_sample = 200`); counter of clipped samples per beam. (Done: `07_emitter/`.)
3. **64-beam test** with the 4000 files distributed over 64 streams (~1 h of stream per beam) and search up to DM 1000 (rank 15, one tree): recall over the files with DM ≤ 1000 (1209), noise FP (/h) and non-FRB FP over all of them, real time and GPU, CPU and RAM use. (Done: section 10 and `docs/RESULTS_64beams.md`.)
4. **RFI:** repeat with the new model when the measurements arrive. (Done: `05_site_rfi/`.)
5. **pirate subbands** (optional): compile `(0,0,0,3,1)` and check whether the B recall rises for narrow-band bursts.

---

## 9. Files

| File at the time | Where it is now |
|---|---|
| `pirate_fil_search.py` (offline search, fp16 direct, `--int4` option) | `03_offline_search/pirate_fil_search.py` |
| `calibrate_snr.py` (S/N calibration with injected pulses) | `03_offline_search/calibrate_snr.py` |
| dedispersion and server configs for CHARTS | `06_configs/` |
| `12_generate_comparison_set_21504.py` (group-3 generator) | `01_simulator/` |
| `validate_21504.py` (visual and S/N validation) | `02_quantization/` |
| `quantization_21504.py` (quantizations measured with the same noise) | `02_quantization/` |
| `quantization_theory.py` (theoretical efficiency η) | `02_quantization/` |
| figure scripts of section 7 | `02_quantization/fig_*.py`, `plot_quantization_theory.py` |

---

## 10. Result: 64-beam real-time test (2026-10-10, without RFI)

**Setup**
- Earlier emitter based on pirate's `FakeXEngine` and a bash launcher; evaluation with `evaluate_test_64b.py`.
- The 4000 group-3 files were distributed over 64 continuous streams of 3510 s each. They go through the full chain: TCP → ring buffer → GPU → grouper → sifter.
- Configuration: 4 bits with n = 2 (centre 32.5); search DM 20–1105, rank 15 with one tree, boxcar ≤ 32 ms.

**Recall + FP with DM ≤ 1000** (297 A, 297 B, 297 negatives and 318 partial):

| Threshold | Noise FP/h (64 beams) | Recall A | Recall B | Recall partial | Non-FRB FP |
|---|---|---|---|---|---|
| S/N ≥ 5 | 1531 | 97.0 % | 86.5 % | 71.4 % | 61.6 % |
| S/N ≥ 6 | 243 | 95.3 % | 84.8 % | 69.5 % | 52.5 % |
| **S/N ≥ 6.54 (10 FP/h)** | **8.9** | **91.9 %** | **82.8 %** | **64.5 %** | **41.1 %** |
| S/N ≥ 7 | 0 | 84.8 % | 81.1 % | 60.1 % | 33.3 % |

**Reading**
- **B:** 98.3 % of the bursts whose expected S/N exceeds the threshold are detected. What is missing in B is almost entirely below the threshold by physics.
- **Partial:** 93.7 % of those exceeding the threshold with their visible part are detected.
- **A** (S/N 8, just above the threshold):
  - 8 % are lost, mostly wide pulses or with scattering ≥ 10 ms at 600 MHz (80 % recall), which exceed the maximum boxcar of 32 ms;
  - the S/N measured by pirate has a median of 8.33, against 8 expected.
- **Noise FP:** they fall very fast with the threshold. At S/N 6.5 they are ~9 per hour summed over the 64 beams (~0.14 per beam and hour).

**Performance**
- The full chain processed at 1.98× real time; the server handles ~126 equivalent beams with a single receiver.
- GPU at 36 % mean use and maximum VRAM of 23.4 GB (of 24; used by the server, the grouper and the emitter's conversion).
- CPU 66 % idle.
- Samples clipped at ±7: 0.126 %, exactly what gaussian noise alone gives, so the bursts are not clipped.
