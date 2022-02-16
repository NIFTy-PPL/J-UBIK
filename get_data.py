import nifty8 as ift
import numpy as np
import sys
import yaml
from lib.observation import ChandraObservationInformation
from lib.output import plot_slices

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 2  # number of log-energy bins
fov = 21.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV // below 2 keV ARF very energy dependant
################################################

with open("obs/obs.yaml", "r") as cfg_file:
    obs_info = yaml.safe_load(cfg_file)

outroot = sys.argv[1]
data_domain = ift.DomainTuple.make(
    [
        ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s),
        ift.RGSpace((npix_e,), distances=np.log(elim[1] / elim[0]) / npix_e),
    ]
)

# retrive data
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
    info = ChandraObservationInformation(
        obs_info["obs" + obsnr], npix_s, npix_e, fov, elim, center
    )
    data = info.get_data(f"./data_{obsnr}.fits")
    data = ift.makeField(data_domain, data)
    plot_slices(data, outroot + f"_data_{obsnr}.png", logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(f"./exposure_{obsnr}")
    exposure = ift.makeField(data_domain, exposure)
    plot_slices(exposure, outroot + f"_exposure_{obsnr}.png", logscale=True)

    # compute the point spread function
    psf_sim = info.get_psf_fromsim(
        (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]), "./psf"
    )
    psf_sim = ift.makeField(data_domain, psf_sim)
    plot_slices(psf_sim, outroot + f"_psfSIM_{obsnr}.png", logscale=False)
    np.save(outfile, {"data": data, "exposure": exposure, "psf_sim": psf_sim})

    if obsnr == obslist[0]:
        center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])

with open("config.yaml", "r") as cfg_file:
    cfg = yaml.safe_load(cfg_file)
    cfg["datasets"] = dataset_list

if cfg:
    with open("config.yaml", "w") as cfg_file:
        yaml.safe_dump(cfg, cfg_file)
