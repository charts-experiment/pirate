# Installation

This folder (`charts/`) lives inside the CHARTS fork of pirate, https://github.com/charts-experiment/pirate:
the folder above **is** pirate. It is cloned once and built in place.

What to install depends on which piece runs on each machine:

| Machine | Runs | Needs |
|---|---|---|
| **Search machine** (with an NVIDIA GPU) | `08_server/` (pirate + grouper + sifter), `09_evaluation/` | the fork built, ksgpu and their conda environment |
| **Machine producing the data** (can be the same) | `07_emitter/` | Python ≥ 3.9 with numpy and PyYAML, the clone of the fork (only `charts/` is used), the data |
| **To generate the test data** (once) | `01_simulator/` | Python 3.12, numpy, scipy and fitburst (`01_simulator/requirements.txt`) |
| **To regenerate the site RFI** (optional) | `05_site_rfi/` 00–07 | numpy, scipy, h5py, matplotlib and the site baseband |

## 1. Repositories and tested versions

| Repository | Address | Tested version | Purpose |
|---|---|---|---|
| **pirate (CHARTS fork)** | https://github.com/charts-experiment/pirate | based on kmsmith137/pirate commit `255edd26c1c77e3f154fc5accd8014f073582b66` (v1.5.0, 2026-09-07) | Network reception and tree dedispersion on the GPU. This repository; `charts/` is the CHARTS code and the rest is unchanged pirate |
| **ksgpu** | https://github.com/kmsmith137/ksgpu | commit `5c81be2dea54226587f80eb206a35ca92c7160c5` (v1.5.0, 2026-09-07) | GPU library used by pirate (required dependency) |
| **fitburst** (CHIME/FRB) | https://github.com/CHIMEFRB/fitburst | commit `3c76da8f9e3ec7bc21951ce1b4a26a0255096b69` | Pulse shape for the simulator (`01_simulator/`) |
| Kotekan (optional) | git@github.com:charts-experiment/kotekan.git | commit `1917da5b` (2021.10-426) | Reference for porting the emitter (`docs/PROTOCOL_pirate_and_Kotekan_adaptation.md`); not needed to run the test |

The original pirate is https://github.com/kmsmith137/pirate. If the fork is updated with newer versions from
kmsmith137 it may work, but the network protocol, the configs and the compiled kernels were tested with 1.5.0.

## 2. Search machine: pirate

### System requirements (not installed by conda)

| | Tested with |
|---|---|
| NVIDIA GPU | RTX 4090, 24 GB (sm_89 architecture) |
| Driver | 590.48.01 |
| System CUDA toolkit (`nvcc`) | 13.1 |
| System compiler | gcc / g++ 13.3 (Ubuntu 24.04) |
| conda (miniconda or miniforge) | any recent one |
| RAM | ≥ 150 GB with `06_configs/test_64b.yml` (64 GB with the optimized config) |
| Disk | ~10 GB for the environment and the build |

### Automatic installation

```bash
git clone https://github.com/charts-experiment/pirate.git && cd pirate
charts/install_pirate.sh /path/to/environment [ARCHITECTURE]
# example: charts/install_pirate.sh ~/envs/pirate 89      (89 = RTX 4090; 80 = A100; 90 = H100)
```

The script **builds the fork it lives in** (it does not download pirate), clones ksgpu next to the fork at the tested
commit, creates the conda environment and builds both. It takes ~5–10 min.

### Manual installation (what the script does)

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/charts-experiment/pirate.git       # the fork: pirate + charts/
git clone https://github.com/kmsmith137/ksgpu  && git -C ksgpu  checkout 5c81be2dea54226587f80eb206a35ca92c7160c5
(cd pirate && git submodule update --init --recursive)          # asdf-cxx

# build only for the GPU you have (makes the build much faster)
echo 'NVCC_ARCH = -gencode arch=compute_89,code=sm_89' > ksgpu/config.mk
echo 'NVCC_ARCH = -gencode arch=compute_89,code=sm_89' > pirate/config.mk

conda env create -p ~/envs/pirate -f pirate/environment_dev.yml
conda activate ~/envs/pirate
conda install -y -c conda-forge clang-format                    # used by the kernel generator, not in the yml

python -m pip install pipmake                                   # ALWAYS use "python -m pip" (see problems below)
(cd ksgpu  && python -m pip install --no-build-isolation -v -e .)
(cd pirate && make asdf-cxx/include/asdf/config.hxx)            # avoids a make -j race
(cd pirate && python -m pip install --no-build-isolation -v -e .)
```

### Check

```bash
conda activate ~/envs/pirate
pirate_frb --help
pirate_frb show dedisperser charts/06_configs/charts_64b_dm1000.yml       # tree memory: host_zones / gpu_zones
pirate_frb time_dedisperser -H -n 24 charts/06_configs/charts_64b_dm1000.yml   # RTX 4090: ~764 beams in real time
```

### Problems found and how they were solved

| Symptom | Cause | Solution |
|---|---|---|
| `pip` installs into Python 2.7 | `~/.local/bin/pip` wins in the PATH | use `python -m pip` inside the environment |
| The kernel generator fails on `clang-format` | `environment_dev.yml` does not include it | `conda install -c conda-forge clang-format` |
| `asdf/config.hxx: No such file` with `make -j` | race in the build | `make asdf-cxx/include/asdf/config.hxx` first |
| `Kernel not found in registry` at run time | the requested combination (rank, boxcar, subbands) is not among the compiled kernels | use the repository configs, or add the row to pirate's `makefile_helper.py` and rebuild |
| `mmap(... MAP_HUGETLB) failed` in `time_dedisperser` | no hugepages configured | add `-H` (or configure `vm.nr_hugepages`) |
| `CUDA out of memory` when the server starts | pirate's GPU pool + grouper + emitter > GPU memory | lower `server.gpu_gib` or `max_gpu_clag` in the YAML |

To update: `git pull` in the fork (and in ksgpu if it changes) and `make -j` in each, with the environment active (the
installation is editable; the pip step does not need to be repeated).

## 3. Machine producing the data: the emitter

```bash
git clone https://github.com/charts-experiment/pirate.git && cd pirate/charts
python3 -m pip install -r 07_emitter/requirements.txt              # numpy and PyYAML
# optional, to quantize or inject RFI on the GPU (gpu: true in the YAML): cupy for the machine's CUDA version
```

It needs neither building pirate nor a GPU: it only uses the `charts/` folder. On the search machine itself, pirate's
environment can be used (it already has numpy, PyYAML and cupy).

## 4. Data (not in the repository)

| Data | Size | Where it comes from | Variable |
|---|---|---|---|
| Group 3: 4000 .fil files of 21 504 channels (A, B, negatives, partial) with `manifest.csv` | ~4.3 TB | `01_simulator/12_generate_comparison_set_21504.py` (see section 4b) | `CHARTS_SET_DIR` |
| Site RFI model `site_rfi_model.npz` | 177 MB | `05_site_rfi/01_dynamic_spectrum.py` + `02_rfi_model.py` from the baseband | `CHARTS_RFI_MODEL` |
| Site baseband (2026-09-02) | 25 GB | capture from the CHARTS X-engine | `CHARTS_BASEBAND_DIR` |

For a quick test without the full set, `--beams-with-data 1 --limit 2` uses only 8 files.

## 4b. Generating the test data

```bash
python3.12 -m venv ~/envs/simulator && source ~/envs/01_simulator/bin/activate
python -m pip install -r 01_simulator/requirements.txt
cd simulator
python 12_generate_comparison_set_21504.py --out /path/group3 --workers 24          # 4000 files, ~4.3 TB
python 12_generate_comparison_set_21504.py --out /path/test --limit 2               # 8 files for a test
```

It is deterministic: it generates the same files used in the tests (verified byte for byte). See `01_simulator/README.md`.

## 5. First run

```bash
cd pirate/charts                                  # everything is run from charts/
conda activate ~/envs/pirate                      # pirate is autodetected (or export CHARTS_PIRATE_ENV=~/envs/pirate)
export CHARTS_SET_DIR=/path/to/group3
python 08_server/launcher.py local --config 06_configs/test_64b.yml \
    --out results/first --limit 2 --processes 16
cat results/first/results.md results/first/alerts.md
```

If pirate or the data folder is missing, the launcher says so before starting. With 8 files and 64 beams it takes
~3–6 min. Expected: 100 % completeness for the FRBs with DM ≤ 1000 in those files, and FRB alerts for the brightest ones.
