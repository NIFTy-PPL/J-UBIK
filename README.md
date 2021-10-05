# Chandra

This Repository contains scripts for data preprocessing, and analysis with numerical information field theory.(NIFTy)
Also the instrument exposure and psf are simulated by [Ciao](https://cxc.cfa.harvard.edu/ciao/) and used for the further analysis.

## Observations
Either get the data [here](https://cda.harvard.edu/chaser/) or via `download_chandra_obsid` as described [here](https://cxc.cfa.harvard.edu/ciao/threads/archivedownload/). The information about the location, obsID, etc. of you data should be stored in /obs/obsID.py.

`obs11713 = {\

            # 1. specify input data
            #######################

            'obsID'         : 11713,
            'data_location' : 'data/11713/repro_20210131/',
            'event_file'    : 'acisf11713_repro_evt2.fits',
            'aspect_sol'    : 'pcadf11713_repro_asol1.fits',
            'bpix_file'     : 'acisf11713_000N002_bpix1.fits',
            'mask_file'     : 'acisf11713_000N002_msk1.fits'
`

### 
