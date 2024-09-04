# J-UBIK

*J*ifty *U*niversal *B*ayesian *I*maging *K*it for photon count instruments is a python package for data analysis of modern X-ray telescopes as Chandra, XMM-Newton and eROSITA.

## Requirements
- [NIFTy8](https://gitlab.mpcdf.mpg.de/ift/nifty) 
- JAX
- astropy
- ciao (>4.14)
- marx (with ciao)
- matplotlib
- [ducc0](https://pypi.org/project/ducc0/)


## Installation of Dependencies
- Information on how to install NIFTy8 can be found [here](https://gitlab.mpcdf.mpg.de/ift/nifty)
- how to install [ciao](https://cxc.cfa.harvard.edu/ciao4.14/download/ciao_install.html)
- how to install [marx](https://cxc.cfa.harvard.edu/ciao/ahelp/install_marx.html)
- set an alias in your .bashrc to source ciao and marx easily

## Installation
This package can be installed via pip. 

    git clone git@gitlab.mpcdf.mpg.de:ift/chandra.git
    git clone https://gitlab.mpcdf.mpg.de/ift/j-ubik
    cd j-ubik
    pip install --user .

for a regular installation. For editable installation add the `-e` flag. 


## Additional Files
Additional calibration files might be needed for instrument-specific pipelines.

### Chandra

### eROSITA
J-UBIK allows to image eROSITA observations with Bayesian posterior uncertainties
and component separation.

#### Requirements
In order to process eROSITA observations and produce realistic synthetic data,
you will need to install [eSASS](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/), the eROSITA
Science Analysis Software System. In particular, the current version of J-UBIK
works with the Docker image version.
Moreover, to compute the eROSITA response accurately, the [caldb](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_CALDB/) folder 
from data release 1 (DR1) or from the early data release (EDR) should be present 
inside the `data/` directory. 
This folder can be downloaded at [caldb download](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/caldb4DR1.tgz).

#### Demo
In the `demo/` repository, `erosita_inference.py` allows to run a generic 
image reconstruction with real and synthetic (mock) eROSITA data.
In order to run a mock demo, you will need to download both the calibration
folder as specified in the Requirements section and an actual observation,
in order to build realistic exposure maps.
A good example is [LMC_dataset](https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/LMC_SN1987A.tar.gz).
For more information on how to run `erosita_demo.py` see the corresponding docstring.

### JWST
