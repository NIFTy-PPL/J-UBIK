import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ciao_contrib.runtool as rt
import nifty8 as ift
import xubik0 as xu

obs_info = xu.get_cfg("obs/obs.yaml")
img_cfg = xu.get_cfg("config.yaml")
grid = img_cfg["grid"]

npix_s = grid["npix_s"] # number of spacial bins per axis
npix_e = grid["npix_e"]  # number of log-energy bins
fov = grid["fov"]  # FOV in arcmin
elim = grid["elim"]  # energy range in keV

outroot = "psf_patches/"
data_domain = xu.get_data_domain(grid)
psf_domain = ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s)

obslist = [
    "14423",
    "9107",
    "13738",
    "13737",
    "13739",
    "13740",
    "13741",
    "13742",
    "13743",
    "14424",
    "14435",
]
center = None

for obsnr in obslist:
    outfile = outroot + f"{obsnr}_" + "patches.npy"
    info = xu.ChandraObservationInformation(obs_info["obs"+obsnr], **grid, center=center)
    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])
    n = 8
    ebin = 0
    psf_sim = xu.get_psfpatches(info, n, npix_s, ebin, fov, num_rays=10e6, Roll=True)
    np.save(outfile, {"psf_sim": psf_sim})
    outname = outroot + f"{obsnr}_"
    xu.plot_psfset(outfile, outname, 1024, 8)
