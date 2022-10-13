import os
import sys

import numpy as np

import nifty8 as ift
import xubik0 as xu

obs_info = xu.get_cfg("obs/obs.yaml")
cfg = xu.get_cfg("scripts/config.yaml")
grid = cfg["grid"]
outroot = cfg["prefix"]
obslist = cfg["datasets"]
grid = img_cfg["grid"]
outroot = img_cfg["outroot"]+img_cfg["prefix"]
if not os.path.exists(outroot):
    os.makedirs(outroot)
obs_type = img_cfg["type"]
if obs_type not in ['CMF', 'EMF', 'SF']:
    obs_type = None
data_domain = xu.get_data_domain(grid)
obslist = img_cfg["datasets"]
center = None
for obsnr in obslist:
    info = xu.ChandraObservationInformation(obs_info["obs" + str(obsnr)], **grid, center=center)
    outfile = outroot + f"_{obsnr}_" + "observation.npy"
    dataset_list.append(outfile)
    info = xu.ChandraObservationInformation(obs_info["obs" + obsnr], **grid, center=center, obs_type=obs_type)
    data = info.get_data(f"../npdata/data_{obsnr}.fits")
    data = ift.makeField(data_domain, data)
    #FIXME info.get_data could also directly return a field, ChandraObservationInformation probably builds the same domain within
    xu.plot_slices(data, outroot + f"_data_{obsnr}.png", logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(f"./exposure_{obsnr}")
    exposure = ift.makeField(data_domain, exposure)
    xu.plot_slices(exposure, outroot + f"_exposure_{obsnr}.png", logscale=True)

    # compute the point spread function
    # psf_sim = info.get_psf_fromsim(
    #     (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]), "./psf", num_rays=cfg["psf_sim"]['num_rays'])
    # psf_sim = ift.makeField(data_domain, psf_sim)
    # xu.plot_slices(psf_sim, outroot + f"_psfSIM_{obsnr}.png", logscale=False)
    np.save(outfile, {"data": data, "exposure": exposure})#, "psf_sim": psf_sim})

    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])
