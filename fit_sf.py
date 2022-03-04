import numpy as np
import matplotlib.pylab as plt
import yaml

import nifty8 as ift
import xubik0 as xu
from xubik.src.library.special_distributions import InverseGammaOperator

ift.set_nthreads(2)

cfg = xu.get_cfg("config.yaml")

npix_s = 1024  # number of spacial bins per axis
fov = 21.0
energy_bin = 0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

#Model P(s)
diffuse = ift.SimpleCorrelatedField(position_space, **cfg['priors_diffuse'])
pspec = diffuse.power_spectrum
diffuse = diffuse.exp()
points = InverseGammaOperator(position_space, **cfg['points'])
signal = points + diffuse
signal = signal.real
signal_dt = signal.ducktape_left('full_signal')

#Likelihood P(d|s)
signal_fa = ift.FieldAdapter(signal_dt.target['full_signal'], 'full_signal')
likelihood_list = []
for dataset in cfg['datasets']:
    #Loop
    observation = np.load(dataset, allow_pickle=True).item()

    #PSF
    psf_arr = observation['psf_sim'].val[:, :, energy_bin]
    psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
    psf_field = ift.Field.from_raw(position_space, psf_arr)
    norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
    psf = norm(psf_field)

    #Data
    data = observation["data"].val[:, :, energy_bin]
    data_field = ift.Field.from_raw(position_space, data)

    #Exp
    exp = observation["exposure"].val[:, :, energy_bin]
    exp_field = ift.Field.from_raw(position_space, exp)
    if dataset == cfg['datasets'][0]:
        norm_first_data = xu.get_norm(exp_field, data_field)
    normed_exp_field = ift.Field.from_raw(position_space, exp) * norm_first_data
    normed_exposure = ift.makeOp(normed_exp_field)

    #Mask
    mask = xu.get_mask_operator(normed_exp_field)

    #Likelihood
    convolved = xu.convolve_field_operator(psf, signal_fa)
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
    print(s)

global_it = cfg['global_it']
n_samples = cfg['Nsamples']
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
        "power_spectrum": pspec,
    },
    output_directory="df_rec",
    initial_position=pos,
    comm=xu.library.mpi.comm,
    inspect_callback=callback,
    overwrite=True,
    resume=True
)

