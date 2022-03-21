import numpy as np
import nifty8 as ift
import xubik0 as xu

obs_info = xu.get_cfg("obs/obs.yaml")
cfg = xu.get_cfg("config.yaml")
grid = cfg["grid"]
outroot = cfg["prefix"]
obslist = cfg["datasets"]

data_domain = xu.get_data_domain(grid)
center = None
for obsnr in obslist:
    outfile = outroot + f"_{obsnr}_" + "observation.npy"
    info = xu.ChandraObservationInformation(obs_info["obs" + obsnr], **grid, center=center)
    data = info.get_data(f"./data_{obsnr}.fits")
    data = ift.makeField(data_domain, data)
    #FIXME info.get_data could also directly return a field, ChandraObservationInformation probably builds the same domain within
    xu.plot_slices(data, outroot + f"_data_{obsnr}.png", logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(f"./exposure_{obsnr}")
    exposure = ift.makeField(data_domain, exposure)
    xu.plot_slices(exposure, outroot + f"_exposure_{obsnr}.png", logscale=True)

    # compute the point spread function
    print(cfg["psf_sim"]['num_rays'])
    psf_sim = info.get_psf_fromsim(
        (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]), "./psf", num_rays=cfg["psf_sim"]['num_rays'])
    psf_sim = ift.makeField(data_domain, psf_sim)
    xu.plot_slices(psf_sim, outroot + f"_psfSIM_{obsnr}.png", logscale=False)
    np.save(outfile, {"data": data, "exposure": exposure, "psf_sim": psf_sim})

    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])
