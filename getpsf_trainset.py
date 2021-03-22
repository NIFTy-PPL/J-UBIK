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
outroot = 'trainset_'
data_domain = ift.DomainTuple.make([ift.RGSpace((npix_s, npix_s), distances=2.*fov/npix_s),\
                                    ift.RGSpace((npix_e,), distances=np.log(elim[1]/elim[0])/npix_e)])

info     = ChandraObservationInformation(obs11713, npix_s, npix_e, fov, elim, center=None)


dx = dy = 0.5 /60
fov_deg = 3. / 60
n_i = int(2* fov_deg / dx) 
n_l = int(2* fov_deg / dy)

zero_loc = (info.obsInfo['aim_ra'] - fov_deg, info.obsInfo['aim_dec'] - fov_deg) 

#for i in range(n_i):
#    for l in range(n_l):
tmp_psf_sim = info.get_psf_fromsim((zero_loc[0] + (16*dy),  zero_loc[1]+ (8*dx)), 'ACIS-I', './psf')
psf_sim  = ift.makeField(data_domain, tmp_psf_sim)
exit()

plot_slices(psf_sim, outroot+'_psfSIM.png', logscale=True)

np.save(outroot+'psf.npy', {'psf_sim': psf_sim})

exit()
