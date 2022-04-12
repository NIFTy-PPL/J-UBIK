import nifty8 as ift
import yaml
import xubik0 as xu
import numpy as np

cfg = xu.get_cfg("config_mf.yaml")

npix_s = 1024  # number of spacial bins per axis
npix_e = 2
npix_freq = npix_e*2
fov = 21.0
elim = (2.0, 10.0)

################## Spaces ###########################
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])
position_space_flattened = ift.RGSpace([position_space.size], distances=[2.0 * fov / npix_s])
padded_energy_space = ift.RGSpace([npix_freq], distances=np.log(elim[1] / elim[0]) / npix_e)
padded_sky_space = ift.DomainTuple.make([position_space, padded_energy_space])
padded_sky_space_flattened = ift.DomainTuple.make([position_space_flattened, padded_energy_space])

energy_space = ift.RGSpace([npix_e], distances=padded_energy_space.distances)
sky_space = ift.DomainTuple.make([position_space, energy_space])
sky_space_flattened = ift.DomainTuple.make([position_space_flattened, energy_space])
#ZeroPadder
depad_sky = ift.FieldZeroPadder(sky_space, new_shape=padded_energy_space.shape, space=1).adjoint

#Limits on Exp
exp_max = np.log(np.finfo(np.float64).max)/2.
exp_min = np.log(np.finfo(np.float64).tiny)


################## Prior ################################
# Diffuse Model
cf_maker = ift.CorrelatedFieldMaker('diffuse')
cf_maker.add_fluctuations(position_space, **cfg['priors_diffuse'])
cf_maker.add_fluctuations(padded_energy_space, **cfg['priors_diffuse_energy'])
cf_maker.set_amplitude_total_offset(**cfg['priors_diffuse_offset'])
diffuse = cf_maker.finalize()
diffuse = depad_sky(diffuse).clip(exp_min, exp_max).exp()

#Points Model

# Log-linear slope part
def transform_loglog_slope_pars(slope_pars):
    """The slope parameters given in the config are
    for slopes in log10/log10 space.
    However, since the energy bins are log10-spaced
    and the signal is modeled in ln-space, the parameters
    have to be transformed prior to their use."""
    res = slope_pars.copy()
    res['mean'] = (res['mean'] + 1) * np.log(10)
    res['sigma'] *= np.log(10)
    return res

ps_spectra_loglinear_slopes = ift.NormalTransform(N_copies=position_space.size, **cfg['points_loglinear_slope'])
unit_slope = np.arange(-energy_space.total_volume/2., energy_space.total_volume/2., energy_space.distances[0])
unit_slope = ift.makeField(energy_space, unit_slope)

slopeOp= xu.ReverseOuterProduct(ps_spectra_loglinear_slopes.target, unit_slope)
ps_spectra_loglinear_part = slopeOp @ ps_spectra_loglinear_slopes

#Fluctuations around log-linear part
cf_maker= ift.CorrelatedFieldMaker('ps_spectrum_fluctuations', total_N=position_space.size)
cf_maker.add_fluctuations(padded_energy_space, **cfg['points_spectrum_fluctuations'])
cf_maker.set_amplitude_total_offset(**cfg['points_spectrum_cfm'])
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

GR_ps = ift.GeometryRemover(sky_space_flattened, space=0)
Reshaper = ift.DomainChangerAndReshaper(sky_space_flattened, sky_space)
point_spectra_normalized = Reshaper @ GR_ps.adjoint @ point_spectra_normalized_raw

FA = ift.FieldAdapter(position_space, 'ps_brightness_xi').clip(-8, 8)
point_brightness = ift.InverseGammaOperator(position_space, **cfg['points_brightness'])(FA)
f_space_spreader = ift.ContractionOperator(point_spectra_normalized.target,
                                           spaces=1).adjoint

points = point_spectra_normalized * f_space_spreader(point_brightness)
signal = points + diffuse
signal = signal.real
signal_dt = signal.ducktape_left('full_signal')
