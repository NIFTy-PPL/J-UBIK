import nifty7 as ift
import numpy as np

from obs.obs11713 import obs11713
from lib.observation import ChandraObservationInformation

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 256       # number of spacial bins per axis
npix_e =  1        # number of log-energy bins
fov    =  4.       # FOV in arcmin
elim   = (2., 10.) # energy range in keV
################################################
outroot = 'strainset_'
data_domain = ift.DomainTuple.make([ift.RGSpace((npix_s, npix_s), distances=2.*fov/npix_s),\
                                    ift.RGSpace((npix_e,), distances=np.log(elim[1]/elim[0])/npix_e)])

info     = ChandraObservationInformation(obs11713, npix_s, npix_e, fov, elim, center=None)


dx = dy = 3. /60
fov_deg = 3. / 60
#please fix this
n_i = int(2* fov_deg / dx)+1
n_l = int(2* fov_deg / dy)+1

zero_loc = (info.obsInfo['aim_ra'] - fov_deg, info.obsInfo['aim_dec'] - fov_deg) 

psf_sim = []
source = []
for i in range(n_i):
    for l in range(n_l):
        tmp_psf_sim = info.get_psf_fromsim((zero_loc[0] + (l * dy),  zero_loc[1]+ (i *dx)), 'ACIS-I', './psf')
        psf_field = ift.makeField(data_domain, tmp_psf_sim)
        psf_sim.append(psf_field)

        tmp_source = np.zeros(tmp_psf_sim.shape)
        pos = np.unravel_index(np.argmax(tmp_psf_sim, axis=None), tmp_psf_sim.shape)
        tmp_source[pos] = 1
        source_field = ift.makeField(data_domain, tmp_source)
        source.append(source_field)

np.save(outroot+'psf.npy', {'psf_sim': psf_sim, 'source': source})

exit()
