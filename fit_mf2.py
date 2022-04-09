import numpy as np
import matplotlib.pylab as plt
import yaml

import nifty8 as ift
import xubik0 as xu

ift.set_nthreads(2)

cfg = xu.get_cfg("config_mf.yaml")

inferred_inverse_gamma = False

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

#Likelihood P(d|s)
signal_fa = ift.FieldAdapter(signal_dt.target['full_signal'], 'full_signal')
likelihood_list = []

exp_norm_mean, exp_norm_std = xu.get_norm_exposure_patches(cfg['datasets'], position_space, 2)
print(f'Mean of exposure-map-norm: {exp_norm_mean} \nStandard deviation of exposure-map-norm: {exp_norm_std}')
datasets = cfg['datasets']
for dataset in [datasets[0]]:
    #Loop
    observation = np.load(dataset, allow_pickle=True).item()

    #PSF
    psf_arr = observation['psf_sim'].val[:, :, :]
    psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
    psf_field = ift.Field.from_raw(sky_space, psf_arr)
    norm = ift.ScalingOperator(sky_space, psf_field.integrate().val **-1)
    psf = norm(psf_field)

    #Data
    data = observation["data"].val[:, :, :]
    data_field = ift.Field.from_raw(sky_space, data)

    #Exp
    exp = observation["exposure"].val[:, :, :]
    exp_field = ift.Field.from_raw(sky_space, exp)
    normed_exp_field = ift.Field.from_raw(sky_space, exp) * np.mean(exp_norm_mean)
    normed_exposure = ift.makeOp(normed_exp_field)

    #Mask
    mask = xu.get_mask_operator(normed_exp_field)

    #Likelihood
    convolved = xu.convolve_field_operator(psf, signal_fa, space=0)
    signal_response = mask @ normed_exposure @ convolved

    masked_data = mask(data_field)
    likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
    likelihood_list.append(likelihood)

likelihood_sum = likelihood_list[0]
for i in range(1, len(likelihood_list)):
    likelihood_sum = likelihood_sum + likelihood_list[i]

likelihood_sum = likelihood_sum(signal_dt)

# End of Loop
ic_newton = ift.AbsDeltaEnergyController(**cfg['ic_newton'])
ic_sampling = ift.AbsDeltaEnergyController(**cfg['ic_sampling'])
minimizer = ift.NewtonCG(ic_newton)

nl_sampling_minimizer = None
pos = 0.1 * ift.from_random(signal.domain)


transpose = xu.Transposer(signal.target)

def callback(samples):
    s = ift.extra.minisanity(
        masked_data,
        lambda x: ift.makeOp(1 / signal_response(signal_dt)(x)),
        signal_response(signal_dt),
        samples,
    )
global_it = cfg['global_it']
n_samples = cfg['Nsamples']

if inferred_inverse_gamma:
    samples = xu.optimize_kl(
        likelihood_sum,
        global_it,
        n_samples,
        minimizer,
        ic_sampling,
        nl_sampling_minimizer,
        plottable_operators={
            "signal": transpose@signal,
            "point_sources": transpose@points,
            "diffuse": transpose@diffuse,
            "power_spectrum": pspec,
            "inverse_gamma_q": points.q(),
            },
        output_directory="df_rec_inferred",
        initial_position=pos,
        comm=xu.library.mpi.comm,
        inspect_callback=callback,
        overwrite=True,
        resume=True
    )
else:
     samples = ift.optimize_kl(
        likelihood_sum,
        global_it,
        n_samples,
        minimizer,
        ic_sampling,
        nl_sampling_minimizer,
        plottable_operators={
            "signal": transpose@signal,
            "point_sources": transpose@points,
            "diffuse": transpose@diffuse,
            },
        output_directory="df_rec",
        initial_position=pos,
        comm=xu.library.mpi.comm,
        inspect_callback=callback,
        overwrite=True,
        resume=True
    )
