import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ciao_contrib.runtool as rt
import nifty8 as ift
import xubik0 as xu

obs_info = xu.get_cfg("obs/obs.yaml")
obs4952 = obs_info["obs4952"]
obs11713 = obs_info["obs11713"]

img_cfg = xu.get_cfg("config.yaml")
grid = img_cfg["grid"]

npix_s = grid["npix_s"] # number of spacial bins per axis
npix_e = grid["npix_e"]  # number of log-energy bins
fov = grid["fov"]  # FOV in arcmin
elim = grid["elim"]  # energy range in keV

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

if False:
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
psf_sim, sources, pos, coords = xu.get_psfpatches(info, n, npix_s, ebin, fov)
np.save(outroot + "psf.npy", {"psf_sim": psf_sim, "sources": sources})

xu.plot_psfset("patches_psf.npy",1024, 8, False)
