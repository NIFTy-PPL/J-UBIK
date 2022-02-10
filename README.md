# Chandra

This Repository contains scripts for data preprocessing, and analysis with numerical information field theory.(NIFTy)
Also the instrument exposure and psf are simulated by [Ciao](https://cxc.cfa.harvard.edu/ciao/) and used for the further analysis.
## Dependencies
- NIFTy8
- ciao
- marx
- matplotlib

## Observations
Either get the data [here](https://cda.harvard.edu/chaser/) or via `download_chandra_obsid` as described [here](https://cxc.cfa.harvard.edu/ciao/threads/archivedownload/). The information about the location, obsID, etc. of you data should be stored in /obs/obs.yaml.


          obs11713:
            obsID: 11713
            data_location: data/11713/repro_20210131/
            event_file: acisf11713_repro_evt2.fits
            aspect_sol: pcadf11713_repro_asol1.fits
            bpix_file: acisf11713_000N002_bpix1.fits
            mask_file: acisf11713_000N002_msk1.fits
            instrument: ACIS-I

More information about the data processing?

## lib 
This directory contains utility functions, i/o, plotting routines, messages, etc.

## Installation of CIAO
More information on this sections follows
- how to install [ciao](https://cxc.cfa.harvard.edu/ciao4.14/download/ciao_install.html)
- how to install [marx](https://cxc.cfa.harvard.edu/ciao/ahelp/install_marx.html)
- set an alias in your bashrc. to source ciao and marx easily

## to do 
- [x] list of observations
- [ ] documentation (model and parameters)
- [ ] document response
- [ ] energy dependency / reconstruct energy bands separately
- [ ] data fusion
- [ ] start run
