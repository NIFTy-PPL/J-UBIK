#!/usr/bin/env python3
import nifty8 as ift
import xubik0 as xu
import numpy as np

def sf_sky(npix_s,
           fov,
           spatial_diffuse_asperity_mean,
           spatial_diffuse_asperity_std,
           spatial_diffuse_flexibility_mean,
           spatial_diffuse_flexibility_std,
           spatial_diffuse_fluctuations_mean,
           spatial_diffuse_fluctuations_std,
           spatial_diffuse_loglogavgslope_mean,
           spatial_diffuse_loglogavgslope_std,
           diffuse_offset_mean,
           diffuse_offset_std_mean,
           diffuse_offset_std_std,
           points_brightness_alpha,
           points_brightness_q):
    # ################# CONFIG ###########################
    # Spaces
    position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

    # Limits on exp
    exp_max = np.log(np.finfo(np.float64).max)/2.
    exp_min = np.log(np.finfo(np.float64).tiny)

    # ################# Prior ################################
    # Diffuse model
    diffuse = ift.SimpleCorrelatedField(position_space,
                                        offset_mean=diffuse_offset_mean,
                                        offset_std=[diffuse_offset_std_mean,diffuse_offset_std_std],
                                        fluctuations=[spatial_diffuse_fluctuations_mean, spatial_diffuse_fluctuations_std],
                                        flexibility=[spatial_diffuse_flexibility_mean,spatial_diffuse_flexibility_std],
                                        asperity=[spatial_diffuse_asperity_mean, spatial_diffuse_asperity_std],
                                        loglogavgslope=[spatial_diffuse_loglogavgslope_mean,spatial_diffuse_loglogavgslope_std])
    pspec = diffuse.power_spectrum
    diffuse = diffuse.clip(exp_min, exp_max).exp()

    # Points Model
    points = ift.InverseGammaOperator(position_space,
                                      alpha=points_brightness_alpha,
                                      q=points_brightness_q)
    points = points.ducktape("points")

    # Signal Model
    signal = points + diffuse
    signal = signal.real
    return signal
