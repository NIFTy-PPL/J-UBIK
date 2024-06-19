# J-UBIK

*J*ifty *U*niversal *B*ayesian *I*maging *K*it for photon count instruments is a python package for data analysis of modern X-ray telescopes as Chandra, XMM-Newton and eROSITA.

## Requirements
- [NIFTy8](https://gitlab.mpcdf.mpg.de/ift/nifty) 
- JAX
- astropy
- ciao (>4.14)
- marx (with ciao)
- matplotlib


## Installation of Dependencies
- Information on how to install NIFTy8 can be found [here](https://gitlab.mpcdf.mpg.de/ift/nifty)
- how to install [ciao](https://cxc.cfa.harvard.edu/ciao4.14/download/ciao_install.html)
- how to install [marx](https://cxc.cfa.harvard.edu/ciao/ahelp/install_marx.html)
- set an alias in your .bashrc to source ciao and marx easily

### Install JWST Dependencies
<!-- THe gwcs could be potentially removed from the Requirements, such taht UBIK doesn't crash once one doesn't have it.-->
- gwcs 
- webbpsf


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
In order to compute the eROSITA response accurately, 
the [caldb](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_CALDB/) folder from DR1 
should be present inside the `data` directory. 
This folder can be downloaded at [caldb download](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/caldb4DR1.tgz).

### JWST
