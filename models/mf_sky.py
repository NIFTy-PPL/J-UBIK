import nifty8 as ift
import xubik0 as xu
import numpy as np


def mf_sky(npix_s,
           npix_e,
           fov,
           e_lim_min,
           e_lim_max,
           spatial_diffuse_asperity_mean,
           spatial_diffuse_asperity_std,
           spatial_diffuse_flexibility_mean,
           spatial_diffuse_flexibility_std,
           spatial_diffuse_fluctuations_mean,
           spatial_diffuse_fluctuations_std,
           spatial_diffuse_loglogavgslope_mean,
           spatial_diffuse_loglogavgslope_std,
           energy_diffuse_asperity_mean,
           energy_diffuse_asperity_std,
           energy_diffuse_flexibility_mean,
           energy_diffuse_flexibility_std,
           energy_diffuse_fluctuations_mean,
           energy_diffuse_fluctuations_std,
           energy_diffuse_loglogavgslope_mean,
           energy_diffuse_loglogavgslope_std,
           diffuse_offset_mean,
           diffuse_offset_std_mean,
           diffuse_offset_std_std,
           spatial_points_loglinear_slope_mean,
           spatial_points_loglinear_slope_std,
           energy_points_asperity_mean,
           energy_points_asperity_std,
           energy_points_flexibility_mean,
           energy_points_flexibility_std,
           energy_points_fluctuations_mean,
           energy_points_fluctuations_std,
           energy_points_loglogavgslope_mean,
           energy_points_loglogavgslope_std,
           points_offset_mean,
           points_offset_std_mean,
           points_offset_std_std,
           points_brightness_alpha,
           points_brightness_q,
           ):
    # ################# CONFIG ###########################
    # Spaces
    position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])
    position_space_flattened = ift.RGSpace([position_space.size], distances=[2.0 * fov / npix_s])
    padded_energy_space = ift.RGSpace([2*npix_e], distances=[np.log(e_lim_max/ e_lim_min) / npix_e])
    energy_space = ift.RGSpace([npix_e], distances=padded_energy_space.distances)
    sky_space = ift.DomainTuple.make([position_space, energy_space])
    sky_space_flattened = ift.DomainTuple.make([position_space_flattened, energy_space])

    # Zero padder
    space_zero_padder = ift.FieldZeroPadder(sky_space, new_shape=padded_energy_space.shape, space=1)

    # Limits on exp
    exp_max = np.log(np.finfo(np.float64).max)/2.
    exp_min = np.log(np.finfo(np.float64).tiny)

    # ################# Prior ################################
    # Diffuse model
    cf_maker = ift.CorrelatedFieldMaker('diffuse')
    cf_maker.add_fluctuations(position_space,
                              fluctuations=[spatial_diffuse_fluctuations_mean, spatial_diffuse_fluctuations_std],
                              flexibility=[spatial_diffuse_flexibility_mean, spatial_diffuse_flexibility_std],
                              asperity=[spatial_diffuse_asperity_mean, spatial_diffuse_asperity_std],
                              loglogavgslope=[spatial_diffuse_loglogavgslope_mean, spatial_diffuse_loglogavgslope_std],
                              prefix='diffuse_position')
    cf_maker.add_fluctuations(padded_energy_space,
                              fluctuations=[energy_diffuse_fluctuations_mean, energy_diffuse_fluctuations_std],
                              flexibility=[energy_diffuse_flexibility_mean, energy_diffuse_flexibility_std],
                              asperity=[energy_diffuse_asperity_mean, energy_diffuse_asperity_std],
                              loglogavgslope=[energy_diffuse_loglogavgslope_mean, energy_diffuse_loglogavgslope_std],
                              prefix='diffuse_energy')
    cf_maker.set_amplitude_total_offset(offset_mean=diffuse_offset_mean,
                                        offset_std=[diffuse_offset_std_mean, diffuse_offset_std_std])
    diffuse = cf_maker.finalize()
    diffuse = space_zero_padder.adjoint(diffuse).clip(exp_min, exp_max).exp()

    # Points Model
    ps_spectra_loglinear_slopes = ift.NormalTransform(mean=spatial_points_loglinear_slope_mean,
                                                      sigma=spatial_points_loglinear_slope_std,
                                                      key='points_slope',
                                                      N_copies=position_space.size)
    unit_slope = np.arange(-energy_space.total_volume/2., energy_space.total_volume/2., energy_space.distances[0])
    unit_slope = ift.makeField(energy_space, unit_slope)

    slope_op = xu.ReverseOuterProduct(ps_spectra_loglinear_slopes.target, unit_slope)
    ps_spectra_loglinear_part = slope_op @ ps_spectra_loglinear_slopes

    # Fluctuations around log-linear part
    cf_maker = ift.CorrelatedFieldMaker('ps_spectrum_fluctuations', total_N=position_space.size)
    cf_maker.add_fluctuations(padded_energy_space,
                              fluctuations=[energy_points_fluctuations_mean, energy_points_fluctuations_std],
                              flexibility=[energy_points_flexibility_mean, energy_points_flexibility_std],
                              asperity=[energy_points_asperity_mean, energy_points_asperity_std],
                              loglogavgslope=[energy_points_loglogavgslope_mean, energy_points_loglogavgslope_std],
                              prefix='points_energy')
    cf_maker.set_amplitude_total_offset(offset_mean=points_offset_mean,
                                        offset_std=[points_offset_std_mean, points_offset_std_std])
    ps_spectra_fluctuations_part_padded = cf_maker.finalize(prior_info=0)

    depadded_domain = ift.DomainTuple.make((ps_spectra_fluctuations_part_padded.target[0], energy_space))
    depad_ps_spectral_flucts = ift.FieldZeroPadder(depadded_domain,
                                                   new_shape=padded_energy_space.shape,
                                                   space=1).adjoint
    ps_spectra_fluctuations_part= depad_ps_spectral_flucts(ps_spectra_fluctuations_part_padded)

    ps_spectra_raw_log = ps_spectra_loglinear_part + ps_spectra_fluctuations_part
    points_spectra_raw = ps_spectra_raw_log.clip(exp_min, exp_max).exp()

    energy_space_integration = ift.IntegrationOperator(points_spectra_raw.target, spaces=1)
    energy_space_spreader = ift.ContractionOperator(points_spectra_raw.target, spaces=1).adjoint

    point_spectra_norm = energy_space_spreader @ energy_space_integration
    adapt = ift.FieldAdapter(points_spectra_raw.target, 'point_spectra_adapt')
    point_spectra_normalized_raw = (adapt * point_spectra_norm(adapt).reciprocal())(
        adapt.adjoint(points_spectra_raw))

    gr_ps = ift.GeometryRemover(sky_space_flattened, space=0)
    reshaper = ift.DomainChangerAndReshaper(sky_space_flattened, sky_space)
    point_spectra_normalized = reshaper @ gr_ps.adjoint @ point_spectra_normalized_raw

    fa = ift.FieldAdapter(position_space, 'ps_brightness_xi').clip(-8, 8)
    point_brightness = ift.InverseGammaOperator(position_space, alpha=points_brightness_alpha, q=points_brightness_q)(fa)
    f_space_spreader = ift.ContractionOperator(point_spectra_normalized.target,
                                               spaces=1).adjoint

    points = point_spectra_normalized * f_space_spreader(point_brightness)
    signal = points + diffuse
    signal = signal.real
    return signal
