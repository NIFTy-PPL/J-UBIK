import nifty7 as ift
import numpy as np
from lib.output import log_plot

if True:
        dct = np.load('varinf_reconstruction.npy', allow_pickle = True).item()

        # log_plot(dct['signal_rec_mean'], 's_mean.png')
        # log_plot(dct['signal_rec_sigma'], 's_sigma.png')
        log_plot(dct['diffuse'], 'diffuse.png')
        log_plot(dct['data'], 'data.png')
        # log_plot(dct['psf_sim'], 'psf_sim.png')
        # log_plot(dct['psf_fit'], 'psf_fit.png')
        # res = dct['signal_rec_mean']-dct['data']
        # log_plot(ift.abs(res), 'residual_var.png')
else:

        dct = np.load('map_reconstruction.npy', allow_pickle = True).item()

        log_plot(dct['data'], 'data.png')
        log_plot(dct['diffuse'], 'diffuse.png')
        log_plot(dct['residual'], 'residual.png')
