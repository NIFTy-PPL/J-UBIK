## TODOS Joss Paper

### Restructure and Test code

- [ ] response.py (including erosita_response.py) -> Matteo
- [ ] data.py (including erosita_data.py) -> Margret
- [ ] sky_models.py (including mf_sk) -> Vincent
- [x] diagnostics.py -> Margret
- [ ] erosita_observation.py -> Matteo 
- [x] chandra_observation.py -> Margret
- [ ] erosita_likelihood.py 
- [ ] chandra_data.py 
- [ ] chandra_response.py
- [ ] chandra_likelihood.py
- [ ] (JWST -> Julian)
- [ ] plotting (mf_plot.py, plot.py, sugar_plot.py) -> Matteo
- [ ] messages.py 
- [ ] utils.py -> Vincent
- [ ] operators
- [ ] structure/ delete tests

### Demos
- [ ] config folder
- [ ] eROSITA demo
- [ ] Chandra demo
- [ ] (JWST demo)
- [ ] evaluation demo

### Paper writing
 - [ ] Write paper text
 - [ ] Include mock example from the view of different instruments
 - [ ] generate homepage

### Afterwards
 - [ ] License
 - [ ] Github accounts
 - [ ] Fix memory state
 - [ ] create git repo with paper.md inside

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
In order to compute the eROSITA response accurately, 
the [caldb](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_CALDB/) folder from DR1 
should be present inside the `data` directory. 
This folder can be downloaded at [caldb download](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/caldb4DR1.tgz).

### JWST
