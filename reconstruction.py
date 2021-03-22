import nifty7 as ift
import numpy as np
import sys

from obs.obs11713 import obs11713
from lib.observation import ChandraObservationInformation
from lib.output import plot_slices
#NOTE this src clashes with nifty7 somehow

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 256       # number of spacial bins per axis
npix_e =  4        # number of log-energy bins
fov    =  4.       # FOV in arcmin
elim   = (2., 10.) # energy range in keV
################################################

outroot = sys.argv[1]
data_domain = ift.DomainTuple.make([ift.RGSpace((npix_s, npix_s), distances=2.*fov/npix_s),\
                                    ift.RGSpace((npix_e,), distances=np.log(elim[1]/elim[0])/npix_e)])

info     = ChandraObservationInformation(obs11713, npix_s, npix_e, fov, elim, center=None)

# retrive data
#data     = info.get_data('./data.fits')
#data     = ift.makeField(data_domain, data)
#plot_slices(data, outroot+'_data.png', logscale=True)

# compute the exposure map
#exposure = info.get_exposure('./exposure')
#exposure = ift.makeField(data_domain, exposure)
#plot_slices(exposure, outroot+'_exposure.png', logscale=True)

psf_ra = (3 + 19/60 + 48.1 / 3600)* 15
psf_dec = 41 + 30/60 + 42/3600
# simulate the PSF for one location
#psf_sim  = info.get_psf_fromsim( (info.obsInfo['aim_ra'], info.obsInfo['aim_dec']), 'ACIS-I', './psf')
psf_sim = info.get_psf_fromsim((49.8770+ 3.5/60,  41.6287+ 3.5/60), 'ACIS-I', './psf')
psf_sim  = ift.makeField(data_domain, psf_sim)
plot_slices(psf_sim, outroot+'_psfSIM.png', logscale=True)

np.save(outroot+'observation.npy', {'data':data, 'exposure':exposure, 'psf_sim':psf_sim})

exit()
