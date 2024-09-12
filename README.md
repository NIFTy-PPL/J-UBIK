# J-UBIK

*J*ifty *U*niversal *B*ayesian *I*maging *K*it for photon count instruments is a python package for data analysis of modern telescopes such as Chandra and eROSITA (X-ray), and JWST (infrared).

J-UBIK allows to image observations from different instruments with Bayesian posterior uncertainties and component separation.


## Requirements
- [NIFTy8](https://gitlab.mpcdf.mpg.de/ift/nifty) 
- JAX
- astropy
- matplotlib
- [ducc0](https://pypi.org/project/ducc0/)


## Installation of Dependencies
- Information on how to install NIFTy8 can be found [here](https://gitlab.mpcdf.mpg.de/ift/nifty)
- Depeding on the instrument you want to use, consider the requirements for chandra / eROSITA / JWST data below.

## Installation
This package can be installed via pip. 

    git clone https://gitlab.mpcdf.mpg.de/ift/j-ubik
    cd j-ubik
    pip install --user .

for a regular installation. For editable installation add the `-e` flag. 


## Additional Files
Additional calibration files might be needed for instrument-specific pipelines.

### Chandra
J-UBIK allows to process observations from chandra x-ray observatory.

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
In order to process eROSITA observations and produce realistic synthetic data,
you will need to:
- Get [eSASS](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/), the eROSITA
Science Analysis Software System. 
In particular, the current version of J-UBIK only supports using eSASS through the 
official docker container to ensure cross-compatibility.
- Download the [caldb](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_CALDB/) folder, this allows to compute the eROSITA response accurately. 
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

### JWST
J-UBIK allows to process and image event files from the James Webb Space Telescope.

#### Requirements
In order to make use of the JWST capabilities of the package, you will need to:
- Install the [jwst](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html) package.
- Install [WebbPSF](https://webbpsf.readthedocs.io/en/stable/installation.html).
- Install [gwcs](https://gwcs.readthedocs.io/en/latest/#installation).\\
For more details see `jwst_demo.py` in the `demo/` repository.

**NOTE**: WebbPSF has shown some compatibility issues with the `numexpr` package.
The current version of the code has been tested successfully on `numexpr version==2.8.4`.