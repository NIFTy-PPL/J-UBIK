import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ciao_contrib.runtool as rt
import nifty8 as ift
import xubik0 as xu

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 4  # number of log-energy bins
fov = 4.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV
################################################

obs_info = xu.get_cfg("obs/obs.yaml")
obs4952 = obs_info["obs4952"]
obs11713 = obs_info["obs11713"]
outroot = "patches_"
data_domain = ift.DomainTuple.make(
    [
        ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s),
        ift.RGSpace((npix_e,), distances=np.log(elim[1] / elim[0]) / npix_e),
    ]
)
psf_domain = ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s)

info = xu.ChandraObservationInformation(obs4952, npix_s, npix_e, fov, elim, center=None)
ref = (
    info.obsInfo["aim_ra"],
    info.obsInfo["aim_dec"],
)  # Center of the FOV in sky coords

if True:
    info = xu.ChandraObservationInformation(
        obs11713,
        npix_s,
        npix_e,
        fov,
        elim,
        center=ref,
    )
n = 8
ebin = 0
psf_sim = xu.get_psfpatches(info, n, npix_s, ebin, fov)
np.save(outroot + "psf.npy", {"psf_sim": psf_sim})
