import nifty8 as ift
import numpy as np

from obs.obs4952 import obs4952
from obs.obs11713 import obs11713
from lib.observation import ChandraObservationInformation

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 1  # number of log-energy bins
fov = 4.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV
################################################
outroot = "patches_"
data_domain = ift.DomainTuple.make(
    [
        ift.RGSpace((npix_s, npix_s), distances=2.0 * fov / npix_s),
        ift.RGSpace((npix_e,), distances=np.log(elim[1] / elim[0]) / npix_e),
    ]
)

info = ChandraObservationInformation(obs4952, npix_s, npix_e, fov, elim, center=None)
ref = (
    info.obsInfo["aim_ra"],
    info.obsInfo["aim_dec"],
)  # Center of the FOV in sky coords
info = ChandraObservationInformation(
    obs11713,
    npix_s,
    npix_e,
    fov,
    elim,
    center=(info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]),
)

# number of patches in one direction. sqrt(N) with N being the the total number of patches
n = 8
fov_deg = 4.0 / 60  # HALF fov in deg
x_0 = ref[0] - fov_deg  # calc origin
y_0 = ref[1] - fov_deg  # calc origin
dx = dy = fov_deg * 2 / n  # dx in sky coordinates
x_i = x_0 + dx / 2
y_i = y_0 + dx / 2
psf_sim = []
source = []
for i in range(n):
    for l in range(n):
        tmp_psf_sim = info.get_psf_fromsim(
            (x_i + (l * dx), y_i + (i * dy)), outroot="./psf", num_rays=1e5
        )
        psf_field = ift.makeField(data_domain, tmp_psf_sim)
        psf_sim.append(psf_field)

        tmp_source = np.zeros(tmp_psf_sim.shape)
        pos = np.unravel_index(np.argmax(tmp_psf_sim, axis=None), tmp_psf_sim.shape)
        tmp_source[pos] = 1
        source_field = ift.makeField(data_domain, tmp_source)
        source.append(source_field)

np.save(outroot + "psf.npy", {"psf_sim": psf_sim, "source": source})
