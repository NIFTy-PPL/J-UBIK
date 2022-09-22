import nifty8 as ift
import xubik0 as xu
import numpy as np

################## CONFIG ###########################
cfg = xu.get_cfg("config_mf.yaml")
npix_s = cfg['grid']['npix_s']  # number of spacial bins per axis
npix_e = cfg['grid']['npix_e']
fov = cfg['grid']['fov']
elim = cfg['grid']['elim']

# Spaces
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])
position_space_flattened = ift.RGSpace([position_space.size], distances=[2.0 * fov / npix_s])
padded_energy_space = ift.RGSpace([2*npix_e], distances=np.log(elim[1] / elim[0]) / npix_e)
padded_sky_space = ift.DomainTuple.make([position_space, padded_energy_space])
energy_space = ift.RGSpace([npix_e], distances=padded_energy_space.distances)
sky_space = ift.DomainTuple.make([position_space, energy_space])
sky_space_flattened = ift.DomainTuple.make([position_space_flattened, energy_space])

# Zero padder
space_zero_padder = ift.FieldZeroPadder(sky_space, new_shape=padded_energy_space.shape, space=1)

# Limits on exp
exp_max = np.log(np.finfo(np.float64).max)/2.
exp_min = np.log(np.finfo(np.float64).tiny)


################## Prior ################################
# Diffuse model
cf_maker = ift.CorrelatedFieldMaker('diffuse')
cf_maker.add_fluctuations(position_space, **cfg['priors_diffuse'])
cf_maker.add_fluctuations(padded_energy_space, **cfg['priors_diffuse_energy'])
cf_maker.set_amplitude_total_offset(**cfg['priors_diffuse_offset'])
diffuse = cf_maker.finalize()
diffuse = space_zero_padder.adjoint(diffuse).clip(exp_min, exp_max).exp()

# Points Model
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
