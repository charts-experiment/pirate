# FRB simulator: the 21 504-channel test set (group 3)

Generates the 4000 `.fil` files used in the 64-beam test: 1000 per set.

| Set | Folder | What it is |
|---|---|---|
| A | `A_snr_fijo` | FRBs with expected S/N 8 (the flux is adjusted to each pulse) |
| B | `B_amplitud_fija` | the same FRBs with a fixed flux of 616.87 Jy (the S/N follows from each one's physics) |
| NEG | `negativos` | pairs of A with dispersion index ≠ −2 (non-FRB) |
| P | `parcial` | FRBs with S/N 15 cut by the edge of the file (5–95 % of the energy visible) |

The folder names (and the `manifest.csv` columns) are the ones the generator writes; they are kept unchanged so that
the existing dataset remains readable and the generator reproduces it byte for byte.

Physics (from `07_generate_comparison_set.py`, function `model()`): gaussian pulse with energy-conserving
scattering, spectrum taken from real bursts in the CHIME catalog (shapes from `catalogs/`), dispersion with smearing
within each channel and ×4 oversampling. Noise of 256 antennas (SEFD from `engine/charts_sefd.py`). DM uniform between
100 and 3000. No RFI (the RFI is added when emitting, see `05_site_rfi/`).

Format: 21 504 channels of 9.389 kHz (672 coarse × 32), 299.85–501.75 MHz, 1 ms, 8 bits (floor 32, σ = 4 counts),
SIGPROC header with channels in descending order. Variable length (7–98 s). One `manifest.csv` per set with the
parameters of each file and `t_top_s` = arrival at 501.6 MHz.

## Files

| File | What it does |
|---|---|
| `12_generate_comparison_set_21504.py` | **The group-3 generator.** Resumable (`_done.jsonl`), parallel |
| `07_generate_comparison_set.py` | Pulse physics and utilities; also generates group 2 (672 channels) |
| `03_convert_to_fil.py` | Writing the `.fil` files and the 8-bit scale |
| `engine/charts_sefd.py`, `engine/fitburst_sim.py` | CHARTS SEFD and fitburst constants/pulse shape |
| `catalogs/` | CHIME/FRB catalogs (clean Catalog 1 and Catalog 2): width, scattering and spectrum come from them |
| `group2_drawn_parameters.csv` | The 2000 draws (1000 main + 1000 partial) reused by group 3 |
| `reconstruct_draws.py` | How the previous CSV was obtained from the group-3 `manifest.csv` files |

Group 3 reuses the group-2 draws. The original draw file is on an external disk; `group2_drawn_parameters.csv` was
reconstructed from the group-3 manifests and verified to regenerate the files **byte for byte** (A_000 and
P_p000_inicio, 2026-11-15).

## Usage

```bash
python 12_generate_comparison_set_21504.py --out /path/group3 --workers 24           # all 4000 (~4.3 TB)
python 12_generate_comparison_set_21504.py --out /path/test --limit 2 --sets A,B     # a few for a test
```

Then: `export CHARTS_SET_DIR=/path/group3` for the emitter.

## Requirements

Python 3.12 (tested), numpy 2.5, scipy 1.18 and **fitburst** from CHIME/FRB:
`pip install -r requirements.txt` (installs fitburst from GitHub at the tested commit).
