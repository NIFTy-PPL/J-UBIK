import os
import numpy as np

import nifty8 as ift
import jubik0 as ju

# load configs as dictionaries from yaml
# obs.yaml contains the info about the observation
# config.yaml contains the information about the inference, e.g.
# binning/resolution/Fov and information about the prior

obs_info = ju.get_config("obs.yaml")
img_cfg = ju.get_config("chandra_cfg.yaml")

grid = img_cfg["grid"]
outroot = img_cfg["paths"]["data_outroot"]

if not os.path.exists(outroot):
    os.makedirs(outroot)

data_domain = ju.get_data_domain(grid)
obslist = img_cfg["datasets"]
center = None

for obsnr in obslist:
    info = ju.ChandraObservationInformation(obs_info[f"obs{obsnr}"],
                                            **grid,
                                            center=center)
    # retrieve data from observation
    data = info.get_data(os.path.join(outroot, f"data_{obsnr}.fits"))
    data = ift.makeField(data_domain, data)
    ju.plot_slices(data, os.path.join(outroot, f"data_{obsnr}.png"), logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(os.path.join(outroot, f"exposure_{obsnr}"))
    exposure = ift.makeField(data_domain, exposure)
    ju.plot_slices(exposure, os.path.join(outroot, f"exposure_{obsnr}.png"), logscale=True)

    # compute the point spread function
    psf_sim = info.get_psf_fromsim((info.obsInfo["aim_ra"],
                                    info.obsInfo["aim_dec"]),
                                   "./psf",
                                   num_rays=img_cfg["psf"]['num_rays'])
    psf_sim = ift.makeField(data_domain, psf_sim)
    ju.plot_slices(psf_sim, os.path.join(outroot, f"psfSIM_{obsnr}.png"), logscale=False)

    # Save the retrieved data
    outfile = os.path.join(outroot, f"{obsnr}_" + "observation.npy")
    np.save(outfile, {"data": data, "exposure": exposure, "psf_sim": psf_sim})

    # Set a center only for the first observation in the list. Keep the center
    # for the other observations
    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])
