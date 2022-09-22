import numpy as np
import matplotlib.pylab as plt
import yaml

import nifty8 as ift
import xubik0 as xu

ift.set_nthreads(2)

with open('models/mf_sky.py', 'r') as fd:
    exec(fd.read())

#Likelihood P(d|s)
signal_fa = ift.FieldAdapter(signal_dt.target['full_signal'], 'full_signal')
likelihood_list = []

exp_norm_max, exp_norm_mean, exp_norm_std = xu.get_norm_exposure_patches(cfg['datasets'], position_space, npix_e)
print(f'Max of exposure-map-norm: {exp_norm_max} \n Mean of exposure-map-norm: {exp_norm_mean} \nStandard deviation of exposure-map-norm: {exp_norm_std}')
for dataset in cfg['datasets']:
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
pos = 0.1 * ift.from_random(signal.domain)



nl_sampling_minimizer = ift.NewtonCG(
        ift.GradientNormController(iteration_limit=3, name='Mini'))
global_it = cfg['global_it']
n_samples = cfg['Nsamples']

samples = xu.optimize_kl(
    likelihood_sum,
    global_it,
    n_samples,
    minimizer,
    ic_sampling,
    nl_sampling_minimizer,
    plottable_operators={
        "signal": signal,
        "point_sources": points,
        "diffuse": diffuse,
        },
    output_directory="df_rec",
    initial_position=pos,
    comm=xu.library.mpi.comm,
    #inspect_callback=callback,
    overwrite=True,
    resume=True,
    return_final_position=True,
    obs_type="CMF"
)
