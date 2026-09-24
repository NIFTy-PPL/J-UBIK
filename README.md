# J-UBIK
The **J**AX-accelerated **U**niversal **B**ayesian **I**maging **K**it  is a python package for high-fidelity Bayesian imaging.

J-UBIK allows to image observations from different instruments with Bayesian posterior uncertainties and component separation.
Next to many useful generic tools and building blocks, JUBIK comes with a series of sky models and instrument implementations, namely:

 - Chandra
 - eROSITA
 - James Webb Space Telescope
 - Gaia (alignment stars for JWST)
 - RESOLVE (radio interferometry)

# Installation
This package can be installed via pip. 

    git clone https://github.com/NIFTy-PPL/J-UBIK
    cd j-ubik
    pip install --user .

for a regular installation. For editable installation add the `-e` flag. 

## Basic requirements

pip installs these with the package:

- [JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [NIFTy](https://gitlab.mpcdf.mpg.de/ift/nifty) (NIFTy.re, the JAX implementation)
- [numpy](https://numpy.org) and [scipy](https://scipy.org)
- [astropy](https://www.astropy.org)
- [ducc0](https://pypi.org/project/ducc0/)
- [matplotlib](https://matplotlib.org)
- [pyyaml](https://pyyaml.org)

## Instrument extras

There is one extra per instrument backend, so a plain install stays lightweight.
Install only what the instrument you work on needs:

    pip install --user .[jwst]      # gwcs, jwst, stpsf, jax-finufft
    pip install --user .[gaia]      # astroquery, for the alignment star search
    pip install --user .[resolve]   # jaxbind, jax-finufft, python-casacore, ehtim
    pip install --user .[healpix]   # jaxbind, for the HEALPix sky model
    pip install --user .[all]       # all of the above

Chandra and eROSITA have no extra. Their instrument software is not on PyPI,
see the sections below.

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

Install the CUDA build of jax and J-UBIK runs on the GPU:

```bash
pip install --user "jax[cuda12]"   # or jax[cuda13], depending on the driver
```

One exception: the PyPI wheel of `jax-finufft` is CPU only. If you want to run
RESOLVE on GPU please read the instructions below.


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
Guidelines for contributing can be found in [CONTRIBUTING.md](CONTRIBUTING.md)

# Instrument requirements
Every instrument section lists what it needs beyond the basic requirements.
Python packages come with the named extra, the rest has to be installed by hand.

---

# Chandra
J-UBIK allows to process observations from the Chandra x-ray observatory.

## Requirements
- [ciao](https://cxc.cfa.harvard.edu/ciao/) >= 4.16
- [marx](https://space.mit.edu/cxc/marx/)

Both are conda-only, so there is no `chandra` extra. Install them via
conda-forge, see [ciao & marx](https://cxc.cfa.harvard.edu/ciao/download/conda.html).

## Demo
`demos/chandra_demo.py` and `demos/chandra_likelihood_demo.py`.

---

# eROSITA
J-UBIK allows to process and image event files from the eROSITA x-ray observatory.

## Requirements
- [eSASS](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/), the
  eROSITA Science Analysis Software System. J-UBIK drives eSASS only through the
  official docker container.
- [caldb](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_CALDB/), the
  calibration folder from data release 1 (DR1) or the early data release (EDR),
  placed inside the `data/` directory. Download:
  [caldb4DR1.tgz](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/caldb4DR1.tgz).
- Public eROSITA data, see [edr](https://erosita.mpe.mpg.de/edr/index.php) and
  [dr1](https://erosita.mpe.mpg.de/dr1/index.html).

There is no `erosita` extra, none of this comes from PyPI.

## Demo
`demos/erosita_demo.py` runs a generic image reconstruction with real and
synthetic (mock) eROSITA data. A mock run needs the calibration folder and an
actual observation to build realistic exposure maps, for example the
[LMC dataset](https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/LMC_SN1987A.tar.gz).
See the docstring of the demo for details.

---

# James Webb Space Telescope
J-UBIK allows to process and image observations from the James Webb Space Telescope.

## Requirements
- [jwst](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html),
  the JWST calibration pipeline
- [stpsf](https://stpsf.readthedocs.io/en/latest/installation.html), the PSF
  model. Its data files are downloaded separately and pointed to by
  `webbpsf_path` in the config.
- [gwcs](https://gwcs.readthedocs.io/en/latest/#installation)
- [jax-finufft](https://pypi.org/project/jax-finufft/) for the nufft rotation
- [astroquery](https://astroquery.readthedocs.io) for the Gaia alignment star
  search, in the separate `gaia` extra

```bash
pip install --user .[jwst,gaia]
```

## Demo
`demos/jwst_demo.py`.

---

# RESOLVE
J-UBIK allows to process and image radio interferometric data.

## Requirements
- [jaxbind](https://pypi.org/project/jaxbind/) for the ducc0 wgridder response
- [jax-finufft](https://pypi.org/project/jax-finufft/) for the finufft
  response, on the GPU see [GPU](#gpu)
- [python-casacore](https://pypi.org/project/python-casacore/) to read CASA
  measurement sets
- [ehtim](https://pypi.org/project/ehtim/) to read uvfits files
- [dask-ms](https://pypi.org/project/dask-ms/) to read zarr datasets, not part
  of the extra
- [casatools and casatasks](https://pypi.org/project/casatasks/) for
  `mstransform` and `statwt` in the measurement-set readout demo, not part of
  the extra

```bash
pip install --user .[resolve]
```

## Demo
`demos/resolve_demo.py`, `demos/resolve_synthetic_demo.py` and
`demos/ms_readout_demo.py`.

## GPU 

The RESOLVE finufft response and the JWST nufft rotation fail at JIT time with
"no lowering for cuda platform" until the package is rebuilt with CUDA. That 
needs `nvcc` on the `PATH`, then:

```bash
CMAKE_ARGS="-DJAX_FINUFFT_USE_CUDA=ON" pip install --force-reinstall --no-deps --no-binary jax-finufft jax-finufft
```

A project that uses J-UBIK with uv can make this the default of every
`uv sync` in its own `pyproject.toml`:

```toml
[tool.uv]
no-binary-package = ["jax-finufft"]

[tool.uv.extra-build-variables]
jax-finufft = { CMAKE_ARGS = "-DJAX_FINUFFT_USE_CUDA=ON" }
```

Verify with:

```bash
python -c "import jax; print(jax.devices())"                     # [CudaDevice(id=0)]
python -c "from jax_finufft import jax_finufft_gpu; print('ok')"
```

The ducc0 wgridder response stays on the CPU, use the finufft backend for GPU
runs.

---

**NOTE**: Importing `jubik` sets the floating point precision in jax to `float64`.
