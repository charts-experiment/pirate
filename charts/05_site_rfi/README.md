# RFI measured at the site, for the 21 504-channel simulations

Created on 2026-10-29. For group 3 it replaces the earlier RFI model v2 (`lineas10`), which came from a 1-antenna
spectrometer at 100 ms. Here the RFI comes from real site voltages, processed at the same resolution as the
simulations (21 504 channels of 9.375 kHz, 1 ms).

## Input data

`260902T161623Z_CHARTS_hdf5/` (25 GB; `export CHARTS_BASEBAND_DIR=...` for 00 and 01). Baseband capture of 2026-09-02.

| | |
|---|---|
| Shape | 32 inputs × 672 channels of 0.3 MHz (300.0–501.6 MHz) × 1 228 800 samples |
| Sampling | 3.333 µs (= 1/0.3 MHz); duration 4.096 s |
| Format | `uint8` = one complex sample: 4 real bits (high nibble) + 4 imaginary (low), two's complement |
| Files | 8 groups of 84 channels × 7 chunks; `baseband_virtual.h5` joins them (32, 672, 1228800) |

**Only inputs 0–7 carry signal** (gaussian noise, kurtosis 2.0, σ from 0.7 to 1.5 levels per component).
Inputs 8–31 are quantized almost only at 0 and ±1 (σ ≈ 0.4 levels, 74 % zeros): they carry no useful signal.
See `input_diagnostics.png`.

Another measurement on the same disk, not used here: `corr_10_24s/` (visibilities every 10.24 s for 24 h from
2026-10-01). It shows how the persistent lines vary over the day.

## Scripts (in order)

| Script | What it does | Output |
|---|---|---|
| `00_input_diagnostics.py` | Power, kurtosis and quantization per input | `input_diagnostics.png` |
| `01_dynamic_spectrum.py` | 32-point FFT per coarse channel → 21 504 channels; \|x\|²; exact 1 ms bins (9.375 frames of 106.67 µs, fractional split). Inputs 0–7 separately | `dynamic_spectrum.npy` (8 × 4096 × 21504, float32, 2.8 GB), `frequencies_mhz.npy` |
| `02_rfi_model.py` | Floor per input, r(t,f) of the beam, persistent and fast RFI, statistics | `site_rfi_model.npz`, `rfi_summary.json`, `site_rfi_model.png` |
| `site_rfi.py` | `SiteRfi` library (numpy or cupy) that adds the RFI to a .fil block | — |
| `sigproc.py` | Reads the header of a .fil (used by 03 and 06) | — |
| `03_fig_before_after.py` | Figure of one file without/with RFI | `B_277_without_vs_with_site_rfi.png` |
| `04_write_fil.py` | The baseband channelized ×32 and integrated to 1 ms, as .fil: per input (antenna); unnormalized float32 and calibrated 8 bits (floor 32, σ = 4 per channel) | `fil/site_260902_input{0..7}_{f32,8bit}.fil` |
| `05_fig_upchannelization.py` | Before (672 channels of 0.3 MHz) and after (21 504 of 9.375 kHz) the channelization, both at 1 ms, one figure per antenna | `figures_upchannelization/antenna{0..7}.png` |
| `06_fig_frb_with_rfi.py` | Simulated FRB with and without the site RFI | `*_frb_with_rfi.png` |
| `07_channel_mask.py` | Static channel masks m1 and m2 from the model | `mask_m1.npy`, `mask_m2.npy` |

Environment: numpy, scipy, h5py and matplotlib (`requirements.txt`) for 00–07; the emitter's (cupy optional) to inject in real time.

Processing check: in clean channels σ/mean = 0.327 at 1 ms, equal to 1/√(9.375 kHz × 1 ms).

## Model

Per input a: `floor_a(f) = B_a(c)·S_a(k)` (B: level of the coarse channel without RFI, iterative running median;
S: PFB scalloping within the coarse channel, 0.63 at the edges to 1.15 at the centre) and
`r_a(t,f) = P_a(t,f)/floor_a(f) − 1`, the RFI power in units of the system power.

Beam: `r = mean of r_a`. For a beam outside the main lobe (sidelobes with random phases) the RFI/system ratio is on
average that of one antenna, so the default attenuation is 0 dB.

- **Persistent** `r_p(f)`: median over the 4.096 s.
- **Fast** `e(t,f)`: excess over r_p larger than 5 σ (σ = temporal MAD per channel), plus connected neighbours > 3 σ.

### Results (`rfi_summary.json`)

| | |
|---|---|
| Channels with lines > 10 dB | 2.1 % (457 channels) |
| Channels with r_p > 1 (> 3 dB) | 12.2 % |
| Worst zones | 410–432 MHz (S/N retained 0.46 after calibration) and 300–340 MHz (0.68) |
| S/N retained for a flat spectrum (calibration) | 0.79 |
| Pixels with fast RFI | 0.49 % |
| Fast events | ~22 000 /s over the whole band; 1 ms median (p99 8 ms); ~19 kHz wide (p99 0.12 MHz) |
| Broadband impulses (> 10 % of the band in 1 ms) | none in 4.1 s |
| External fast RFI (coincides in another input) | 40–70 % in inputs 0, 4, 5, 6, 7; ~1 % in 1 and 2 (local) |
| Saturated at 255 after injection | 0.68 % |

There are also a train of periodic pulses at ~452 MHz (every ~85 ms), a line that appears after 2.4 s at
400 MHz and modulation at 362 MHz.

## How it is added to a .fil (`site_rfi.py`)

With s = x − 32 (signal + noise of σ0 = 4 counts), P_sys = 4·√9.375 = 12.25 counts, g = 10^(−attenuation/10),
rp = g·r_p(f), re = g·e(t,f):

- **Visible** channel (line with 10 log10(1 + rp) > `line_threshold_db`, 10 dB by default as in RFI v2):
  `y = 32 + s + P_sys·(rp + re) + σ0·√((1 + rp + re)² − 1)·N(0,1)`
- **Calibrated** channel (the per-channel offset/scale of the 8-bit stage normalizes the persistent power):
  `y = 32 + [s + P_sys·re + σ0·√((1 + rp + re)² − 1)·N(0,1)] / (1 + rp)`

The result is rounded and clipped to 0–255. It is the same physics as RFI v2.

Files longer than 4.096 s: bootstrap of contiguous 4.096 s blocks with a random offset within the capture
(deterministic per seed and block).

## Usage

**Real time, without writing to disk** (`07_emitter/emitter.py --rfi`):
```
python 08_server/launcher.py local --config 06_configs/test_64b.yml --out results/rfi --rfi --mask 05_site_rfi/mask_m1.npy
```
All beams use the same RFI clock (it reaches all of them at once through the sidelobes). The `gain` argument of
`SiteRfi.apply` gives each beam a different sidelobe gain. On the GPU it costs ~10 ms per block and beam
(0.6 s per 2.048 s chunk with 64 beams) and 176 MB.

## Limitations

- **A single 4.1 s capture.** In long files the RFI repeats in blocks; there is no variation on scales of minutes or
  hours, nor broadband impulses (none appeared in 4.1 s). More baseband is needed, or `corr_10_24s/` could be used to
  modulate the persistent lines over the day.
- **4+4 bits with σ ≈ 1 level:** the strongest lines (> ~15 dB) saturate at ±7 and their power is underestimated.
- **8 antennas without beamforming:** the beam is assumed to see the RFI as the average of the antennas (average
  sidelobe). Beams with a null or a lobe towards the source would see less or more; this can be varied with
  `attenuation_db` and `gain`.
- **Frequency grid:** the capture uses 0.3 MHz per coarse channel and the simulations 0.30045 MHz. Each simulated
  channel takes the measured channel nearest in frequency, and 300.0 MHz is assumed to be the centre of coarse channel 0.

## First tests (2026-10-29)

**Offline, B_277 with `pirate_fil_search.py`** (normalizes each channel per chunk):

| | 8 bits | 4 bits (n = 2) |
|---|---|---|
| FRB S/N without → with RFI | 30.7 → 27.9 | 26.3 → 23.4 |
| Candidates S/N ≥ 5 in 18 s, without → with RFI | 155 → 1763 | 158 → 1087 |

The RFI produces candidates at DM 44–121 with S/N 11–14.

**Real time, 64 beams, short test (80 files):**

| Run | At 10 FP/h: threshold | Recall A / B / partial | Typical maximum per chunk |
|---|---|---|---|
| no RFI | 6.54 | 100 / 83 / 50 % | S/N ~5.8 |
| no RFI, per-channel normalization | 6.68 | 100 / 83 / 50 % | S/N ~5.9 |
| site RFI + normalization | not reached (1116 FP/h at S/N 5–8) | 0 / 33 / 0 % | S/N ~20, 48 ms, DM ~78 |

- The emitter's fixed quantization does not work with RFI: saturated channels add an offset and every chunk gives
  S/N ≈ 118. That is why `--rfi` switches to per-channel quantization (median and σ per channel and chunk;
  saturated → −8).
- Even when normalizing, RFI that varies within the chunk gives S/N ~20 in wide boxcars at low DM, and the toy grouper
  (one event per chunk and beam) always delivers that maximum. RFI mitigation was missing (masking channels by
  variance/kurtosis, or a stage-1 classifier) as well as a grouper with several events per chunk. Both came later:
  mask m1 and grouper v1 (see `docs/RESULTS_64beams.md`).

## Band-wide level variations (not included in the injection model)

In the 8-bit .fil there are level jumps over the whole band at once (~1.5, 2.1, 2.7 and 3.5 s). In the 6457 clean
channels (r_p < 0.05), the band average at 1 ms fluctuates more than the noise (expected σ 0.0124 per channel):
×2.3–3.1 in inputs 0, 1, 2, 5 and 6; ×4.7–7.5 in 3, 4 and 7.
They are variations common to all channels, of ~0.03–0.09 σ per channel: each pixel stays below the 5σ threshold, so
`e(t,f)` does not capture them and the current injection does not reproduce them. Summed over the band they weigh a lot
at DM ≈ 0 and low DM with wide boxcars. Pending: add a common term r_bb(t) to the model.
