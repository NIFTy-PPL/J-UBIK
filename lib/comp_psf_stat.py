import nifty8 as ift
import numpy as np
import sys

from obs.obs4952 import obs4952
from obs.obs11713 import obs11713
from lib.observation import ChandraObservationInformation
from lib.output import plot_slices

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024  # number of spacial bins per axis
npix_e = 4  # number of log-energy bins
fov = 4.0  # FOV in arcmin
elim = (2.0, 10.0)  # energy range in keV
################################################

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

# psf_sim = info.get_psf_fromsim((psf_ra, psf_dec), "./psf",num_rays=1e4)
# psf_sim = ift.makeField(data_domain, psf_sim)
# plot_slices(psf_sim, "psfSIM_1e4.png", logscale=True)
# np.save("psf_1e4.npy", psf_sim)

psf_sim = info.get_psf_fromsim((psf_ra, psf_dec), "./psf", num_rays=1e6)
psf_sim = ift.makeField(data_domain, psf_sim)
plot_slices(psf_sim, "psfSIM_1e6_evenmoreflux.png", logscale=True)
np.save("psf_1e6_evenmoreflux.npy", psf_sim)

# psf_sim = info.get_psf_fromsim((psf_ra, psf_dec), "./psf",num_rays=1e7)
# psf_sim = ift.makeField(data_domain, psf_sim)
# plot_slices(psf_sim, "psfSIM_1e7.png", logscale=True)
# np.save("psf_1e7.npy", psf_sim)
