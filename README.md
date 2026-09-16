# J-UBIK
The **J**AX-accelerated **U**niversal **B**ayesian **I**maging **K**it  is a python package for high-fidelity Bayesian imaging.

J-UBIK allows to image observations from different instruments with Bayesian posterior uncertainties and component separation.
Next to many useful generic tools and building blocks, JUBIK comes with a series of sky models and instrument implementations, namely:

 - Chandra
 - eROSITA
 - James Webb Space Telescope

# Installation
This package can be installed via pip. 

    git clone https://github.com/NIFTy-PPL/J-UBIK
    cd j-ubik
    pip install --user .

for a regular installation. For editable installation add the `-e` flag. 

## Instrument extras

There is one extra per instrument backend, so a plain install stays lightweight.
Install only what the instrument you work on needs:

    pip install --user .[jwst]      # gwcs, jwst, stpsf, jax-finufft
    pip install --user .[gaia]      # astroquery, for the alignment star search
    pip install --user .[resolve]   # jaxbind, jax-finufft, python-casacore, ehtim
    pip install --user .[erosita]   # no pip dependencies, see the eROSITA section
    pip install --user .[all]       # all of the above

Chandra has no extra. CIAO and marx are conda-only, see the Chandra section.

With [uv](https://docs.astral.sh/uv/), the same via the project environment:

```bash
uv sync --extra jwst              # JWST only
uv sync --extra resolve           # RESOLVE only
uv sync --all-extras              # every instrument backend
```

Note that `uv sync` makes the environment match exactly the extras you list, so
passing a single `--extra` removes the packages belonging to the others. Pass
every extra you want in one command, or use `uv sync --all-extras`.

## GPU

J-UBIK runs on the GPU as soon as the CUDA build of jax is installed. There is
no `gpu` extra: an extra is easy to forget, and the next plain `uv sync` or
`uv run` without it silently drops back to CPU jax. Declare the GPU stack in the
project that uses J-UBIK instead, so it is the default of that project's
environment.

Two packages matter:

- `jax[cuda12]` (or `jax[cuda13]`, depending on the driver). The wheel ships the
  CUDA runtime, only the NVIDIA driver has to exist on the machine.
- `jax-finufft`, used by the RESOLVE finufft response and the JWST nufft
  rotation. Its PyPI wheel is CPU only. On the GPU the model fails at JIT time
  with "no lowering for cuda platform" until the package is rebuilt from source
  with CUDA enabled, which needs `nvcc` on the `PATH`.

With uv both go into the downstream `pyproject.toml`:

```toml
dependencies = [
    "jubik[resolve]",
    "jax[cuda12]",
    "jax-finufft",
]

[tool.uv]
# The PyPI wheel is CPU only. Build from the sdist with CUDA on. uv caches the
# built wheel, so only the first sync per machine pays the 10 to 20 minutes.
no-binary-package = ["jax-finufft"]

[tool.uv.extra-build-variables]
# One binary for every GPU the project runs on, here sm 86 (RTX 30xx) and
# sm 90 (H100). Table: https://developer.nvidia.com/cuda-gpus
jax-finufft = { CMAKE_ARGS = "-DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;90" }
```

Verify with:

```bash
python -c "import jax; print(jax.devices())"                     # [CudaDevice(id=0)]
python -c "from jax_finufft import jax_finufft_gpu; print('ok')"
```

With pip, rebuild jax-finufft once after installing `jax[cuda12]`:

```bash
CMAKE_ARGS="-DJAX_FINUFFT_USE_CUDA=ON" pip install --force-reinstall --no-deps --no-binary jax-finufft jax-finufft
```

Without `CMAKE_CUDA_ARCHITECTURES` the build targets the GPU of the machine it
runs on.

The ducc0 wgridder response (`backend: ducc0`) stays on the CPU. jaxbind calls
it as a host callback, so on the GPU every application of the response copies
the sky to the host and back. Use the finufft backend for GPU runs.

## Development

Test and documentation tooling lives in [PEP 735](https://peps.python.org/pep-0735/)
dependency groups, so neither is installed by default:

```bash
uv sync --group dev               # pytest, sphinx, ipython
uv sync --only-group test         # what CI runs
```

With pip (25.1 or newer):

```bash
pip install -e . --group dev
```

# Requirements
- [JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [astropy](https://www.astropy.org)
- [NIFTy](https://gitlab.mpcdf.mpg.de/ift/nifty) (follow installation for NIFTy.re, the JAX implementation of NIFTy)
- [ducc0](https://pypi.org/project/ducc0/)
- [matplotlib](https://matplotlib.org/stable/install/index.html)

# Testing
Testing needs [pytest](https://docs.pytest.org/en/stable/), which the `test`
dependency group provides. To run the tests execute the following from the
`j-ubik` directory:

```bash
uv sync --only-group test   # or: pip install -e . --group test
pytest test/
```

Tests considering Chandra are skipped if `ciao` is not installed.

# Contributing
Guidelines for contribuation can be found in [CONTRIBUTING.md](CONTRIBUTING.md)


# Instrument specific Requirements 
- [Chandra](#chandra)
- [eROSITA](#erosita)
- [James Webb Space Telescope](#james-webb-space-telescope)

# Additional Files
Additional calibration files might be needed for instrument-specific pipelines.

---

# Chandra
J-UBIK allows to process observations from Chandra x-ray observatory.

## Requirements
- ciao >= 4.16
- marx

We recommend installation of both via conda / conda-forge
[ciao & marx](https://cxc.cfa.harvard.edu/ciao/download/conda.html)

---

# eROSITA
J-UBIK allows to process and image event files from the eROSITA x-ray observatory.

## Requirements
To process eROSITA observations or produce realistic synthetic data,
you will need:
- [eSASS](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/), the eROSITA
Science Analysis Software System. 
In particular, the current version of J-UBIK only supports using eSASS through the 
official docker container to ensure cross-compatibility.
- [caldb](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_CALDB/) folder, this allows to compute the eROSITA response accurately. 
Either the caldb from data release 1 (DR1) or from the early data release (EDR) should be present 
inside the `data/` directory. 
This folder can be downloaded at [caldb download](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/caldb4DR1.tgz).
- Download the data if you want to work with public eROSITA data, see [edr](https://erosita.mpe.mpg.de/edr/index.php) and [dr1](https://erosita.mpe.mpg.de/dr1/index.html).  

## Demo
In the `demo/` repository, `erosita_inference.py` allows to run a generic 
image reconstruction with real and synthetic (mock) eROSITA data.
In order to run a mock demo, you will need to download both the calibration
folder as specified in the Requirements section and an actual observation,
in order to build realistic exposure maps.
A good example is [LMC_dataset](https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/LMC_SN1987A.tar.gz).
For more information on how to run `erosita_demo.py` see the corresponding docstring.

---

# James Webb Space Telescope
J-UBIK allows to process and image event files from the James Webb Space Telescope.

## Requirements
In order to make use of the JWST capabilities of the package, you will need to:
- Install the [jwst](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html) package.
- Install [stpsf](https://stpsf.readthedocs.io/en/latest/installation.html).
- Install [gwcs](https://gwcs.readthedocs.io/en/latest/#installation).

For more details see `jwst_demo.py` in the `demo/` repository.
Alternatively, you can install these requirements via 
```bash
pip install --user .[jwst]
```
 
---

# RESOLVE
J-UBIK allows to process and image radio interferometic data.

## Requirements
In order to make use of the RESOLVE capabilities of the package, you will need to:
- Install the [jaxbind](https://pypi.org/project/jaxbind/) to work with the wgridder radio response
- Install the [jax-finufft](https://pypi.org/project/jax-finufft/) to work with the FinuFFT radio response
- Install the [python-casacore](https://pypi.org/project/python-casacore/) to work with CASA measurement sets.
- Install [ehtim](https://pypi.org/project/ehtim/) to read uvfits files.

Alternatively, you can install all of these requirements via
```bash
pip install --user .[resolve]
```

---


**NOTE**:
- Importing `jubik` sets the floating point precision in jax to `float64`. 
- WebbPSF has shown some compatibility issues with the `numexpr` package.  
The current version of the code has been tested successfully on `numexpr version==2.8.4`.
