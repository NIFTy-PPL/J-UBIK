# %% [markdown]
# # Chandra Demo
# The `ChandraObservatioinInformation` Class is an object, carrying the
# nessecary information to interact with the [`CIAO`](https://cxc.cfa.harvard.edu/ciao/)
# interface. This inferace implements convenience functions to get
#  - data 
#  - exposure
#  - psf(s)
# for a certain **observation**, **binning** in space and energy.
#


# %%
import matplotlib.pyplot as plt
import numpy as np

import jubik0 as ju
from os import makedirs

makedirs("chandra_demo_files")

# %% [markdown]
# For this, `J-UBIK` needs the file paths to the most important files,
# from one observation. The files can be retrieved via the CIAO terminal
# interface [`download_chandra_obsid`](https://cxc.cfa.harvard.edu/ciao/ahelp/download_chandra_obsid.html)
# or via the web interface [`chaser`](https://cda.harvard.edu/chaser/).
#
#  Also `J-UBIK` needs some other important information about 
#  - spatial pixels 
#  - spectral pixels
#  - energy limits (elim) 
#  - energy ranges.
# Therefore, we define the dictionary `obsInfo` holding the filepaths
# and the discussed variables.
# 
# %%
obsInfo = {"obsID": 4952,
           "data_location": "../data/4952/",
           "event_file": "primary/acisf04952N004_evt2.fits",
           "aspect_sol": "primary/pcadf04952_000N001_asol1.fits",
           "bpix_file": "primary/acisf04952_000N004_bpix1.fits",
           "mask_file": "secondary/acisf04952_000N004_msk1.fits",
           "instrument": "ACIS-I"}

npix_s = 512
npix_e = 1
fov = 2024
half_fov_arcmin = fov/2/60
energy_ranges = (3, 7.0)
elim =  (3, 7.0)

# %% [markdown]
# With these, we can get an instance of ChandraObservationInformation.
#

# %%
print("The following also gives information about the RA, DEC, ROLL")
print("of the observation as well as the Observation duration")
chandra_obs = ju.ChandraObservationInformation(obsInfo=obsInfo, 
                                               npix_s=npix_s, 
                                               npix_e=npix_e, 
                                               fov=fov, 
                                               elim=elim, 
                                               energy_ranges=energy_ranges)
# %% [markdown]
#
# ## Data and Exposure
# To get the binned data for the observation, we to use the methods `get_data`.

# %%
data = chandra_obs.get_data("test")

# %%
plt.imshow(data[:, :, 0], origin="lower", norm="log", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin])
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.colorbar(label="photon counts")
plt.savefig("chandra_demo_files/ChandraData.png")

# %% [markdown]
# ![](chandra_demo_files/ChandraData.png)

# %% [markdown]
# The same can be done for the exposure. Here we use `get_exposure`. This gives us the exposure in units of seconds and cm squared.

# %%
exposure = chandra_obs.get_exposure("exp-test")

# %%
plt.imshow(exposure[:, :, 0], origin="lower", norm="log", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin])
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.colorbar(label=r"$\mathrm{s} \times \mathrm{cm}^2$")
plt.savefig("chandra_demo_files/ChandraExposure.png")

# %% [markdown]
# ![](chandra_demo_files/ChandraExposure.png)

# %% [markdown]
#
# ## The PSF
# To get the PSF we use MARX. Since we also want to use `J-UBIK` for far
# off-axis signal reconstructions, the morphology of the psf is of importance. We can also specify the number of simulated photons.
# %%
psf = chandra_obs.get_psf_fromsim((chandra_obs.obsInfo["ra"], chandra_obs.obsInfo["dec"]), detector_type="ACIS-I", outroot="psf", num_rays=1e7)
# %%
plt.imshow(psf[:, :, 0], origin="lower", norm="log", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin])
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.colorbar(label=r"photon counts")
plt.savefig("chandra_demo_files/ChandraPSFCenter.png")

# %% [markdown]
# ![](chandra_demo_files/ChandraPSFCenter.png)

# %% [markdown]
# For wide field reconstructions we need more than one PSF. In order to get the information on how the PSF changes over the field of view, we can simulate it for equidistant positions on the detector. Here we already do a normalization of the PSFs. For the plotting we add 1e-5 to remove the extrem contrast between 1 count and no count in the log plotting routine.

# %%
psfs = ju.get_psfpatches(info=chandra_obs,
                        n=8,
                        num_rays=1e6,
                        npix_s=npix_s,
                        ebin=0, 
                        Roll=False)

# %%
psfs_full = psfs.sum(axis=0)
plt.imshow(psfs_full+1e-5, origin="lower", norm="log", interpolation="none", extent=[-half_fov_arcmin, half_fov_arcmin, -half_fov_arcmin, half_fov_arcmin])
plt.xlabel("x in [arcmin]")
plt.ylabel("y in [arcmin]")
plt.colorbar(label=r"photon counts")
plt.savefig("chandra_demo_files/ChandraPSFfov.png")
# %% [markdown]
# ![](chandra_demo_files/ChandraPSFfov.png)
