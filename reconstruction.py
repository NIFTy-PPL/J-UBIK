import nifty8 as ift
import numpy as np
import sys

from obs.obs4952 import obs4952
from obs.obs4948 import obs4948
from obs.obs11713 import obs11713
from lib.observation import ChandraObservationInformation
from lib.output import plot_slices
#NOTE this src clashes with nifty7 somehow

########## RECONSTRUCTION PARAMETERS ##########
npix_s = 1024      # number of spacial bins per axis
npix_e =  4        # number of log-energy bins
fov    =  4.       # FOV in arcmin
elim   = (2., 10.) # energy range in keV #TODO Lower limit? sollte das nicht bis 0.1 keV runter gehen=
################################################

outroot = sys.argv[1]
data_domain = ift.DomainTuple.make([ift.RGSpace((npix_s, npix_s), distances=2.*fov/npix_s),\
                                    ift.RGSpace((npix_e,), distances=np.log(elim[1]/elim[0])/npix_e)])


obses = (obs4952, obs4948)
center = None
for ii, obs in enumerate(obses):
    # retrive data
    info     = ChandraObservationInformation(obs, npix_s, npix_e, fov, elim, center)
    data     = info.get_data(f'./data_{ii}.fits')
    data     = ift.makeField(data_domain, data)
    plot_slices(data, outroot+f'_data_{ii}.png', logscale=True)

    # compute the exposure map
    exposure = info.get_exposure(f'./exposure_{ii}')
    exposure = ift.makeField(data_domain, exposure)
    plot_slices(exposure, outroot+f'_exposure_{ii}.png', logscale=True)

    #TODO FIX THIS MESS
    #psf_ra = (3 + 19/60 + 48.1 / 3600)* 15
    #psf_dec = 41 + 30/60 + 42/3600

    # simulate the PSF for one location

    #psf_sim  = info.get_psf_fromsim( (info.obsInfo['aim_ra'], info.obsInfo['aim_dec']), 'ACIS-I', './psf')
    #psf_sim = info.get_psf_fromsim((49.8770+ 3.5/60,  41.6287+ 3.5/60), 'ACIS-I', './psf')
    #psf_sim  = ift.makeField(data_domain, psf_sim)
    #plot_slices(psf_sim, outroot + f'_psfSIM_{ii}.png', logscale=True)

    np.save(outroot+f'_{ii}_'+'observation.npy', {'data':data, 'exposure':exposure})  #, 'psf_sim':psf_sim})
    center = (info.obsInfo['aim_ra'], info.obsInfo['aim_dec'])
exit()
