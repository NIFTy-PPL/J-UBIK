import nifty6 as ift
import numpy as np
import sys

from obs.obs11713_old_worklaptop import obs11713
# NOTE this _old_worklaptop is not int the repository
from src.observation import ChandraObservationInformation
from src.output import plot_slices

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
data     = info.get_data('./data.fits')
data     = ift.makeField(data_domain, data)
plot_slices(data, outroot+'_data.png', logscale=True)

# compute the exposure map
exposure = info.get_exposure('./exposure')
exposure = ift.makeField(data_domain, exposure)
plot_slices(exposure, outroot+'_exposure.png', logscale=True)

# simulate the PSF for one location
psf_sim  = info.get_psf_fromsim( (info.obsInfo['aim_ra'], info.obsInfo['aim_dec']), 'ACIS-I', './psf')
psf_sim  = ift.makeField(data_domain, psf_sim)
plot_slices(psf_sim, outroot+'_psfSIM.png', logscale=True)

np.save(outroot+'observation.npy', {'data':data, 'exposure':exposure, 'psf_sim':psf_sim})

exit()
