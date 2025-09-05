# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt

import jubik0 as ju

import jax
import jax.numpy as jnp
import numpy as np

from jax import config
config.update('jax_enable_x64', True)


# FIXME rename jubik0 to jubik
# FIXME Why is energy at the end, flip this
# FIXME WTF is wrong with psf
# FIXME not consistent: if one energy bin: no bin for data / exp but for psf?
# -

# # Chandra Demo
#
# The `ChandraObservatioinInformation` Class is an object, carrying the
# nessecary information to interact with the [`CIAO`](https://cxc.cfa.harvard.edu/ciao/)
# interface. This inferace implements convenience functions to get
# - data 
# - exposure
# - psf(s)
#
# for a certain **observation**, **binning** in space and energy.
#
# For this, `J-UBIK` needs the file paths to the most important files,
# from one observation. The files can be retrieved via the CIAO terminal
# interface [`download_chandra_obsid`](https://cxc.cfa.harvard.edu/ciao/ahelp/download_chandra_obsid.html)
# or via the web interface [`chaser`](https://cda.harvard.edu/chaser/).
#
# Also `J-UBIK` needs some other important information about 
# - spatial pixels 
# - spectral pixels
# - energy limits (elim) 
# - energy ranges.
#
# Therefore, we define the dictionary `obsInfo` holding the filepaths
# and the discussed variables.

# +
obsInfo = {"obsID": 4948,
           "data_location": "../data/4948/",
           "event_file": "primary/acisf04948N004_evt2.fits",
           "aspect_sol": "primary/pcadf04948_001N001_asol1.fits",
           "bpix_file": "primary/acisf04948_001N004_bpix1.fits",
           "mask_file": "secondary/acisf04948_001N004_msk1.fits",
           "instrument": "ACIS-I"}

npix_s = 1024
npix_e = 2
fov = 512
energy_ranges = (0.3, 2.0, 3.0)
elim =  (1,5)
# -

# With these, we can get an instance of ChandraObservationInformation.

print("The following also gives information about the RA, DEC, ROLL")
print("of the observation as well as the Observation duration")
chandra_obs = ju.ChandraObservationInformation(obsInfo=obsInfo, 
                                               npix_s=npix_s, 
                                               npix_e=npix_e, 
                                               fov=fov, 
                                               elim=elim, 
                                               energy_ranges=energy_ranges)

# ## Data and Exposure
#
# To get the binned data and exposure for the observation and the binning
# configuration defined, we only need to use the methods `get_data` and `get_exposure`

data = chandra_obs.get_data("test")
# FIXME Why is energy at the end, flip this

fig, ax = plt.subplots(1,2, figsize=(10,3))
ax[0].imshow(data[:,:,0], origin="lower", norm="log", interpolation="none")
im = ax[1].imshow(data[:,:,1], origin="lower", norm="log", interpolation="none")
fig.colorbar(im, ax=ax)

exposure = chandra_obs.get_exposure("exp-test")


fig, ax = plt.subplots(1,2, figsize=(10,3))
ax[0].imshow(exposure[:,:,0], origin="lower", norm="log", interpolation="none")
im = ax[1].imshow(exposure[:,:,1], origin="lower", norm="log", interpolation="none")
fig.colorbar(im, ax=ax)

# ## The PSF
#
# To get the PSF we use MARX. Since we also want to use `J-UBIK` for far
# off-axis signal reconstructions, the morphology of the psf is of importance.

psf = chandra_obs.get_psf_fromsim((chandra_obs.obsInfo["ra"], chandra_obs.obsInfo["dec"]), "psf", num_rays=1e4)

fig, ax = plt.subplots(1,2, figsize=(10,3))
ax[0].imshow(psf[:,:,0], origin="lower", interpolation="none")
im = ax[1].imshow(psf[:,:,1], origin="lower", interpolation="none")
fig.colorbar(im, ax=ax)


