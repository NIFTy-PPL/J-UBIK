import numpy as np
import sys
import yaml

import nifty8 as ift
import xubik0 as xu

obs_info = xu.get_cfg("obs/obs.yaml")
multifrequency = False
if multifrequency:
    img_cfg = xu.get_cfg("config_mf.yaml")
else:
    img_cfg = xu.get_cfg("config.yaml")
grid = img_cfg["grid"]
outroot = img_cfg["prefix"]
obs_type = img_cfg["type"]
if obs_type not in ['CMF', 'EMF', 'SF']:
    obs_type = None
data_domain = xu.get_data_domain(grid)
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
dataset_list = []
for obsnr in obslist:
    outfile = outroot + f"_{obsnr}_" + "observation.npy"
    dataset_list.append(outfile)
    info = xu.ChandraObservationInformation(obs_info["obs" + obsnr], **grid, center=center, obs_type=obs_type)
    data = info.get_data(f"data_{obsnr}.fits")
    data = ift.makeField(data_domain, data)
    #FIXME info.get_data could also directly return a field, ChandraObservationInformation probably builds the same domain within
    xu.plot_slices(data, outroot + f"_data_{obsnr}.png", logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(f"./exposure_{obsnr}")
    exposure = ift.makeField(data_domain, exposure)
    xu.plot_slices(exposure, outroot + f"_exposure_{obsnr}.png", logscale=True)

    # compute the point spread function
    psf_sim = info.get_psf_fromsim(
        (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]), "./psf", num_rays=img_cfg["psf_sim"]['num_rays'])
    psf_sim = ift.makeField(data_domain, psf_sim)
    xu.plot_slices(psf_sim, outroot + f"_psfSIM_{obsnr}.png", logscale=False)
    np.save(outfile, {"data": data, "exposure": exposure, "psf_sim": psf_sim})

    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])

#TODO simplify this stuff
with open("config.yaml", "r") as cfg_file:
    cfg = yaml.safe_load(cfg_file)
    #FIXME: Import yaml, pleaseee!
    cfg["datasets"] = dataset_list

if cfg:
    with open("config.yaml", "w") as cfg_file:
        yaml.safe_dump(cfg, cfg_file)
