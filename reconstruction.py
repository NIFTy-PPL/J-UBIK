import nifty6 as ift
import numpy as np

from obs.obs11713 import obs11713
from src.observation import ChandraObservationInformation

info     = ChandraObservationInformation(obs11713, 256, 16, 4, (1.,10.))
#data     = info.get_data('./data.fits')
#exposure = info.get_exposure('./exposure')
psf_siim = info.get_psf_fromsim( (info.obsInfo['aim_ra'], info.obsInfo['aim_dec']), 'ACIS-I', './psf')
exit()
