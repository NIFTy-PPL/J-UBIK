#!/usr/bin/env bash
# Builds a separate environment for sht_backend="cunusht" (Linux + CUDA 12).
#
# Needs on PATH: python3.12, git, nvcc (CUDA 12 toolkit), cmake>=3.18, a C++17
# compiler and FFTW3 (with fftw3_omp, found through FFTW_HOME if set). On an
# HPC system load the matching modules first, see the README. Usage:
#
#   scripts/cunusht/install.sh <env-dir> [<src-dir>]
#
# Afterwards run the GPU pytest command printed below.
# Validated versions go to <env-dir>/validated-*.txt.
set -euo pipefail

# Absolute paths: the SHTns build below runs from inside its checkout.
mkdir -p "${1:?usage: install.sh <env-dir> [<src-dir>]}" "${2:-$1/src}"
ENV=$(cd "$1" && pwd)
SRC=$(cd "${2:-$1/src}" && pwd)
JUBIK=$(cd "$(dirname "$0")/../.." && pwd)

# Upstream commits this backend was written against.
CUNUSHT_COMMIT=7079a5e    # Sebastian-Belkner/cunuSHT main, 2024-09
JAXBIND_GPU_COMMIT=75d401b  # jknollm/jaxbind-gpu main, 2026-10
SHTNS_COMMIT=4e69ceb      # schaeffn/shtns master (SHTns 3.7.5)

: "${CUDA_PATH:=$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_PATH
# Module systems usually export FFTW_HOME; the rpath lets SHTns find FFTW
# later without modules loaded.
if [ -n "${FFTW_HOME:-}" ]; then
    export SHTNS_LIB_DIR=$FFTW_HOME/lib
    export CPATH=$FFTW_HOME/include${CPATH:+:$CPATH}
    export LIBRARY_PATH=$FFTW_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
    export LDFLAGS="-Wl,-rpath,$FFTW_HOME/lib${LDFLAGS:+ $LDFLAGS}"
fi

python3.12 -m venv "$ENV"
PY="$ENV/bin/python"
"$PY" -m pip install -U pip wheel setuptools
"$PY" -m pip install \
    "jax[cuda12]==0.11.2" "jaxlib==0.11.2" "numpy>=2.1" "cupy-cuda12x>=13" \
    "cufinufft>=2.4,<3" ducc0 healpy lenspyx line_profiler logdecorator \
    nanobind scikit-build-core pybind11 pytest

clone() {  # clone <url> <dir> <commit>
    [ -d "$SRC/$2" ] || git clone -q "$1" "$SRC/$2"
    git -C "$SRC/$2" checkout -q "$3"
}

# SHTns with its CUDA Python module (_shtns_cuda). setup.py only adds it when
# CUDA_PATH is set and silently drops it if it fails, hence the check.
clone https://gricad-gitlab.univ-grenoble-alpes.fr/schaeffn/shtns.git shtns "$SHTNS_COMMIT"
(cd "$SRC/shtns" && git clean -qfdx && "$PY" -m pip install --no-build-isolation .)
"$PY" -c "import shtns; assert shtns._shtns.__name__ == '_shtns_cuda', 'SHTns was built without CUDA'"

clone https://github.com/jknollm/jaxbind-gpu.git jaxbind-gpu "$JAXBIND_GPU_COMMIT"
"$PY" -m pip install --no-build-isolation "$SRC/jaxbind-gpu"

# cunuSHT: native extension, then the package, with jubik's small patch.
clone https://github.com/Sebastian-Belkner/cunuSHT.git cunuSHT "$CUNUSHT_COMMIT"
git -C "$SRC/cunuSHT" checkout -q -- .
git -C "$SRC/cunuSHT" apply "$JUBIK/scripts/cunusht/cunusht.patch"
"$PY" -m pip install --no-build-isolation "$SRC/cunuSHT/cunusht/c"
"$PY" -m pip install --no-build-isolation --no-deps "$SRC/cunuSHT"

"$PY" -m pip install -e "$JUBIK[healpix]"

"$PY" -m pip freeze > "$ENV/validated-freeze.txt"
for d in shtns jaxbind-gpu cunuSHT; do
    echo "$d $(git -C "$SRC/$d" rev-parse HEAD)"
done > "$ENV/validated-commits.txt"
echo "jubik $(git -C "$JUBIK" rev-parse HEAD)" >> "$ENV/validated-commits.txt"
cat "$ENV/validated-commits.txt"
echo "Installed. Next, on a CUDA GPU with no CUDA modules loaded (module purge):"
echo "env -u JAX_PLATFORMS XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -m pytest -v -rs $JUBIK/test/test_cunusht_sht.py $JUBIK/test/sky_model/multifrequency/test_cunusht_mf_sky.py"
