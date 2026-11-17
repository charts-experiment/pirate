#!/bin/bash
# Builds THIS fork of pirate (the folder above charts/) and its ksgpu dependency, with their conda environment.
# Usage: charts/install_pirate.sh ENVIRONMENT_DIR [GPU_ARCHITECTURE] [KSGPU_DIR]
#   GPU_ARCHITECTURE: 89 (RTX 4090, default), 86 (RTX 30xx/A40), 80 (A100), 90 (H100)
#   KSGPU_DIR: where to clone ksgpu (default: next to the fork)
# Requires on the system: conda, git, nvcc (CUDA toolkit) and gcc/g++. See charts/INSTALL.md.
set -euo pipefail
ENVDIR=${1:?folder of the conda environment to create}
ARCH=${2:-89}
PIRATE=$(cd "$(dirname "$0")/.." && pwd)                 # the fork: not downloaded, this folder is built
KSGPU=${3:-$(dirname "$PIRATE")/ksgpu}
KSGPU_COMMIT=5c81be2dea54226587f80eb206a35ca92c7160c5   # the tested version (kmsmith137/ksgpu)
PIRATE_TESTED=255edd26c1c77e3f154fc5accd8014f073582b66  # kmsmith137's pirate 1.5.0, base of the fork

for c in conda git nvcc g++; do command -v $c > /dev/null || { echo "'$c' is missing on the system (see charts/INSTALL.md)"; exit 1; }; done
[ -f "$PIRATE/pirate_frb/__init__.py" ] || { echo "pirate not found in $PIRATE: run it from the clone of the fork"; exit 1; }
git -C "$PIRATE" merge-base --is-ancestor $PIRATE_TESTED HEAD 2>/dev/null \
    || echo "WARNING: this pirate does not contain the tested commit ${PIRATE_TESTED:0:8}; it may not be compatible"

[ -d "$KSGPU" ] || git clone https://github.com/kmsmith137/ksgpu "$KSGPU"
git -C "$KSGPU" checkout $KSGPU_COMMIT
git -C "$PIRATE" submodule update --init --recursive       # asdf-cxx

echo "NVCC_ARCH = -gencode arch=compute_${ARCH},code=sm_${ARCH}" > "$KSGPU/config.mk"
echo "NVCC_ARCH = -gencode arch=compute_${ARCH},code=sm_${ARCH}" > "$PIRATE/config.mk"

source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$ENVDIR" ] || conda env create -p "$ENVDIR" -f "$PIRATE/environment_dev.yml"
conda activate "$ENVDIR"
conda install -y -c conda-forge clang-format

python -m pip install pipmake
(cd "$KSGPU" && python -m pip install --no-build-isolation -v -e .)
(cd "$PIRATE" && make asdf-cxx/include/asdf/config.hxx)
(cd "$PIRATE" && python -m pip install --no-build-isolation -v -e .)

pirate_frb --help > /dev/null && echo "pirate (fork, $(git -C "$PIRATE" rev-parse --short HEAD)) installed in $ENVDIR (sm_$ARCH)"
echo "Next: conda activate $ENVDIR   (see charts/INSTALL.md, section 5)"
