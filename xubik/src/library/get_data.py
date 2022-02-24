import nifty8 as ift
import numpy as np
import sys
import yaml
from ..operators.observation_operator import ChandraObservationInformation
from .plot import plot_slices

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 4  # number of log-energy bins
fov = 4.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV
# TODO Lower limit? sollte das nicht bis 0.1 keV runter gehen
################################################

with open("obs/obs.yaml", 'r') as cfg_file:
    obs_info = yaml.safe_load(cfg_file)

outroot = sys.argv[1]
data_domain = ift.DomainTuple.make(
    [
        ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s),
        ift.RGSpace((npix_e,), distances=np.log(elim[1] / elim[0]) / npix_e),
    ]
)

center = None

# retrive data
info = ChandraObservationInformation(obs_info['obs4952'], npix_s, npix_e, fov, elim, center)
data = info.get_data(f"./data_4952.fits")
data = ift.makeField(data_domain, data)
plot_slices(data, outroot + f"_data_4952.png", logscale=True)

# compute the exposure map
exposure = info.get_exposure(f"./exposure_4952")
exposure = ift.makeField(data_domain, exposure)
plot_slices(exposure, outroot + f"_exposure_4952.png", logscale=True)

# compute the point spread function
psf_sim = info.get_psf_fromsim(
    (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]), "./psf"
)
psf_sim = ift.makeField(data_domain, psf_sim)
plot_slices(psf_sim, outroot + f"_psfSIM_4952.png", logscale=True)

np.save(
    outroot + f"_4952_" + "observation.npy", {"data": data, "exposure": exposure, 'psf_sim':psf_sim})
# center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])

# info = ChandraObservationInformation(obs_info['obs11713'], npix_s, npix_e, fov, elim, center)
# data = info.get_data(f"./data_11713.fits")
# data = ift.makeField(data_domain, data)
# plot_slices(data, outroot + f"_data_11713.png", logscale=True)

# # compute the exposure map
# exposure = info.get_exposure(f"./exposure_11713")
# exposure = ift.makeField(data_domain, exposure)
# plot_slices(exposure, outroot + f"_exposure_11713.png", logscale=True)

# np.save(
#     outroot + f"_11713_" + "observation.npy", {"data": data, "exposure": exposure})  # , 'psf_sim':psf_sim})
