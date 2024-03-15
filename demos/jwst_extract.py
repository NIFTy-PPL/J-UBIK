from jwst import datamodels
import webbpsf


from os import environ
from os.path import join, exists
from os import mkdir
from sys import exit

from charm_lensing.utils import save_fits, make_header
import matplotlib.pyplot as plt

import numpy as np

import yaml

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u


from sys import exit


OUT_PATH = './data/'

FIG_SHAPE = 128  # 96 pixels
WORLD_LOCATION = (64.66543063107049, -47.86462563973049)  # (ra, dec) in deg
WL = SkyCoord(
    WORLD_LOCATION[0] * u.deg, WORLD_LOCATION[1] * u.deg, frame='icrs')
DATA_RANGE = (1, 3)
OVERSAMPLE_LIST = [2, 4]
FILTER = 'F356W'
FILTER = 'F444W'


if FILTER == 'F356W':
    out_dir_name = 'f356w_cal0{data_number}'
    path = '/home/jruestig/Data/jwst/jw01355_SPT0418-47/jw01355-o016_t001_nircam_clear-f356w'
    subfolder = 'jw01355016001_02103_0000{data_number}_nrcblong'
    filename = 'jw01355016001_02103_0000{data_number}_nrcblong_cal.fits'

elif FILTER == 'F444W':
    out_dir_name = 'f444w_cal0{data_number}'
    path = '/home/jruestig/Data/jwst/jw01355_SPT0418-47/jw01355-o016_t001_nircam_clear-f444w'
    subfolder = 'jw01355016001_02105_0000{data_number}_nrcblong'
    filename = 'jw01355016001_02105_0000{data_number}_nrcblong_cal.fits'


for data_number in range(*DATA_RANGE):
    out_dir = out_dir_name.format(data_number=data_number)
    out_dir = join(OUT_PATH, out_dir)
    if not exists(out_dir):
        mkdir(out_dir)

    fin = datamodels.open(join(path,
                               subfolder.format(data_number=data_number),
                               filename.format(data_number=data_number)))

    wtd = fin.meta.wcs.get_transform('world', 'detector')
    dtw = fin.meta.wcs.get_transform('detector', 'world')

    data = fin.data
    err = fin.err
    d_pix = wtd(*WORLD_LOCATION)

    ii0 = int(np.round(d_pix[1])) - FIG_SHAPE//2
    ii1 = int(np.round(d_pix[1])) + FIG_SHAPE//2
    jj0 = int(np.round(d_pix[0])) - FIG_SHAPE//2
    jj1 = int(np.round(d_pix[0])) + FIG_SHAPE//2

    data = data[ii0:ii1, jj0:jj1]
    err = err[ii0:ii1, jj0:jj1]
    mean = data[:20, :20][~np.isnan(data[:20, :20])].mean()
    std = data[:20, :20][~np.isnan(data[:20, :20])].std()

    # # HEADER TRANSFORMATION
    # header['NAXIS1'] = FIG_SHAPE
    # header['NAXIS2'] = FIG_SHAPE
    # header['CRPIX1'] = FIG_SHAPE//2

    print('Building PSF')
    environ["WEBBPSF_PATH"] = '/home/jruestig/Data/jwst/WebbPsf/webbpsf-data_1.2.1/'
    nircam = webbpsf.NIRCam()
    nircam.filter = FILTER
    nircam.detector_position = d_pix

    # save the mean and the std of the first 20 pixels in a text file in the same folder
    with open(join(out_dir, 'mean_std.txt'), 'w') as f:
        f.write(f'Mean[:20, :20] = {mean} \nStd[:20, :20] = {std}')

    pix_shift = np.array((
        d_pix[0] - np.round(d_pix[0]),
        d_pix[1] - np.round(d_pix[1])
    ))
    np.save(join(out_dir, 'pix_shift.npy'), pix_shift)
    save_fits(data, join(out_dir, 'data.fits'))
    save_fits(err,  join(out_dir, 'err.fits'))

    for oversample in OVERSAMPLE_LIST:
        psf = nircam.calc_psf(fov_pixels=FIG_SHAPE, oversample=oversample)
        psf[0].writeto(
            join(out_dir, f'psf_oversample_{oversample}.fits'), overwrite=True)
        psf[1].writeto(join(out_dir, 'psf_oversample_1.fits'), overwrite=True)
