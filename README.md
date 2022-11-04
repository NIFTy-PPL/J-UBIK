# X-UBIK

*X*-ray *U*niversal *B*ayesian *I*maging *K*it is a python package for data analysis of modern X-ray telescopes as Chandra, XMM-Newton and eROSITA

## Requirements
- [NIFTy8](https://gitlab.mpcdf.mpg.de/ift/nifty) 
- astropy
- ciao (>4.13)
- marx
- matplotlib


## Installation of Dependencies
- Information on how to install NIFTy8 can be found [here](https://gitlab.mpcdf.mpg.de/ift/nifty)
- how to install [ciao](https://cxc.cfa.harvard.edu/ciao4.14/download/ciao_install.html)
- how to install [marx](https://cxc.cfa.harvard.edu/ciao/ahelp/install_marx.html)
- set an alias in your .bashrc to source ciao and marx easily

## Installation
This package can be installed via pip. 

    git clone git@gitlab.mpcdf.mpg.de:ift/chandra.git
    cd xubik
    pip install --user .

for a regular installation. For editable installation add the `-e` flag. 

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
