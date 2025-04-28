# J-UBIK
The **J**AX-accelerated **U**niversal **B**ayesian **I**maging **K**it  is a python package for high-fidelity Bayesian imaging.

J-UBIK allows to image observations from different instruments with Bayesian posterior uncertainties and component separation.
Next to many useful generic tools and building blocks, JUBIK comes with a series of sky models and instrument implementations, namely:

 - Chandra
 - eROSITA
 - James Webb Space Telescope

## Requirements
- [JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [astropy](https://www.astropy.org)
- [NIFTy8](https://gitlab.mpcdf.mpg.de/ift/nifty) 
- [ducc0](https://pypi.org/project/ducc0/)
- [matplotlib](https://matplotlib.org/stable/install/index.html)


## Instrument specific Requirements 
- [Chandra](#chandra)
- [eROSITA](#erosita)
- [James Webb Space Telescope](#james-webb-space-telescope)

## Installation
This package can be installed via pip. 

    git clone https://github.com/NIFTy-PPL/J-UBIK
    cd j-ubik
    pip install --user .

for a regular installation. For editable installation add the `-e` flag. 


## Additional Files
Additional calibration files might be needed for instrument-specific pipelines.

### Chandra
J-UBIK allows to process observations from Chandra x-ray observatory.

#### Requirements
- [ciao](https://cxc.cfa.harvard.edu/ciao4.14/download/ciao_install.html) > 4.16
- [marx](https://cxc.cfa.harvard.edu/ciao/ahelp/install_marx.html)
- set an alias in your .bashrc to source ciao and marx easily 

NOTE: in case you install ciao via conda, make sure that all environmental are set
conda env config vars set MARX_ROOT /soft/marx/marx-5.2.0 (or where your marx is installed)
conda env config vars set MARX_DATA_DIR ${MARX_ROOT}/share/marx/data

### eROSITA
J-UBIK allows to process and image event files from the eROSITA x-ray observatory.

#### Requirements
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

#### Demo
In the `demo/` repository, `erosita_inference.py` allows to run a generic 
image reconstruction with real and synthetic (mock) eROSITA data.
In order to run a mock demo, you will need to download both the calibration
folder as specified in the Requirements section and an actual observation,
in order to build realistic exposure maps.
A good example is [LMC_dataset](https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/LMC_SN1987A.tar.gz).
For more information on how to run `erosita_demo.py` see the corresponding docstring.

### James Webb Space Telescope
J-UBIK allows to process and image event files from the James Webb Space Telescope.

#### Requirements
In order to make use of the JWST capabilities of the package, you will need to:
- Install the [jwst](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html) package.
- Install [WebbPSF](https://webbpsf.readthedocs.io/en/stable/installation.html).
- Install [gwcs](https://gwcs.readthedocs.io/en/latest/#installation).\\
For more details see `jwst_demo.py` in the `demo/` repository.

**NOTE**:
- Importing `jubik0` sets the floating point precision in jax to `float64`. 
- WebbPSF has shown some compatibility issues with the `numexpr` package.  
The current version of the code has been tested successfully on `numexpr version==2.8.4`.
