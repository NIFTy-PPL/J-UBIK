import nifty8 as ift
import numpy as np
import sys

from obs.obs4952 import obs4952
from obs.obs11713 import obs11713
from lib.observation import ChandraObservationInformation
from lib.output import plot_slices

# FIXME for the long term put this somewhere else
########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 4  # number of log-energy bins
fov = 4.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV
################################################

outroot = sys.argv[1]
data_domain = ift.DomainTuple.make(
    [
        ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s),
        ift.RGSpace((npix_e,), distances=np.log(elim[1] / elim[0]) / npix_e),
    ]
)

obses = (obs4952, obs11713)
info = ChandraObservationInformation(obses[0], npix_s, npix_e, fov, elim, center=None)

psf_ra = (3 + 19 / 60 + 48.1 / 3600) * 15
psf_dec = 41 + 30 / 60 + 42 / 3600

psf_sim = info.get_psf_fromsim((psf_ra, psf_dec), "ACIS-S", "./psf")
psf_sim = ift.makeField(data_domain, psf_sim)
plot_slices(psf_sim, "psfSIM_ob0.png", logscale=True)
np.save("psf_ob0.npy", psf_sim)

info = ChandraObservationInformation(
    obses[1],
    npix_s,
    npix_e,
    fov,
    elim,
    center=(info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]),
)

psf_sim = info.get_psf_fromsim((psf_ra, psf_dec), "ACIS-I", "./psf")
psf_sim = ift.makeField(data_domain, psf_sim)
plot_slices(psf_sim, "psfSIM_ob1.png", logscale=True)

np.save("psf_ob1.npy", psf_sim)
