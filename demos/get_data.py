import os
import numpy as np

import nifty8 as ift
import xubik0 as xu

# load configs as dictionaries from yaml
# obs.yaml contains the info about the observation
# config.yaml contains the information about the inference, e.g.
# binning/resolution/Fov and information about the prior

obs_info = xu.get_config("obs/obs.yaml")
img_cfg = xu.get_config("scripts/config.yaml")

obslist = img_cfg["datasets"]
grid = img_cfg["grid"]
outroot = img_cfg["outroot"]+img_cfg["prefix"]

if not os.path.exists(outroot):
    os.makedirs(outroot)

data_domain = xu.get_data_domain(grid)
obslist = img_cfg["datasets"]
center = None

for obsnr in obslist:
    info = xu.ChandraObservationInformation(obs_info["obs" + obsnr],
                                            **grid,
                                            center=center)
    # retrieve data from observation
    data = info.get_data(f"../npdata/data_{obsnr}.fits")
    data = ift.makeField(data_domain, data)
    xu.plot_slices(data, outroot + f"_data_{obsnr}.png", logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(f"./exposure_{obsnr}")
    exposure = ift.makeField(data_domain, exposure)
    xu.plot_slices(exposure, outroot + f"_exposure_{obsnr}.png", logscale=True)

    # compute the point spread function
    psf_sim = info.get_psf_fromsim((info.obsInfo["aim_ra"],
                                    info.obsInfo["aim_dec"]),
                                   "./psf",
                                   num_rays=cfg["psf_sim"]['num_rays'])
    psf_sim = ift.makeField(data_domain, psf_sim)
    xu.plot_slices(psf_sim, outroot + f"_psfSIM_{obsnr}.png", logscale=False)

    # Save the retrieved data
    outfile = outroot + f"_{obsnr}_" + "observation.npy"
    np.save(outfile, {"data": data, "exposure": exposure, "psf_sim": psf_sim})

    # Set a center only for the first observation in the list. Keep the center
    # for the other observations
    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])
