import numpy as np
import matplotlib.pyplot as plt
import nifty8 as ift

import jubik0 as ju

ift.set_nthreads(2)

# Load config file
cfg = ju.get_config("scripts/config.yaml")
prefix = cfg["prefix"]
npix_s = cfg["grid"]["npix_s"]
fov = cfg["grid"]["fov"]

energy_bin = 0
position_space = ift.RGSpace([npix_s, npix_s], distances=[fov / npix_s])

# Model P(s)
diffuse = ift.SimpleCorrelatedField(position_space, **cfg['priors_diffuse'])
pspec = diffuse.power_spectrum
diffuse = diffuse.exp()

points = ift.InverseGammaOperator(position_space, **cfg['points'])
points = points.ducktape("points")

signal = points + diffuse
signal = signal.real
signal_dt = signal.ducktape_left('full_signal')

# Likelihood P(d|s)
signal_fa = ift.FieldAdapter(signal_dt.target['full_signal'], 'full_signal')
likelihood_list = []

for dataset in cfg['datasets']:
    # Loop
    observation = np.load("data/npdata/"+prefix+"_"+str(dataset)+"_observation.npy",
                          allow_pickle=True).item()

    # PSF
    psf_file = np.load("data/npdata/psf_patches/"+str(dataset)+"_patches.npy",
                       allow_pickle=True).item()["psf_sim"]
    psfs = []
    for p in psf_file:
        psfs.append(p.val)
    psfs = np.array(psfs, dtype="float64")

    # Data
    data = observation["data"].val[:, :, energy_bin]
    data_field = ift.Field.from_raw(position_space, data)
    # Likelihood
    conv_op = ju.OAnew.cut_force(signal_fa.target, psfs, 64, 16)
    convolved = conv_op @ signal_fa
    cut = ju.MarginZeroPadder(convolved.target,
                              ((position_space.shape[0] - convolved.target.shape[0])//2),
                              space=0).adjoint

    # Exp
    exp = observation["exposure"].val[:, :, energy_bin]
    exp_field = ift.Field.from_raw(position_space, exp)

    cut_exp_field = cut(exp_field)
    exposure_op = ift.makeOp(cut_exp_field)

    # Mask
    mask_op = ju.get_mask_operator(cut_exp_field)

    # Signal Response
    signal_response = mask_op @ exposure_op @ convolved

    # Prepare Data
    cut_data = cut(data_field)
    masked_data = mask_op(cut_data)

    # build likelihood
    likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
    likelihood.name = dataset
    likelihood_list.append(likelihood)

likelihood_sum = likelihood_list[0]
for i in range(1, len(likelihood_list)):
    likelihood_sum = likelihood_sum + likelihood_list[i]
likelihood_sum = likelihood_sum(signal_dt)

# End of Loop
ic_newton = ift.AbsDeltaEnergyController(**cfg['ic_newton'])
ic_sampling = ift.AbsDeltaEnergyController(**cfg['ic_sampling'])
minimizer = ift.NewtonCG(ic_newton)

# Initial Position for inference
pos = 0.1 * ift.from_random(signal.domain)
transpose = ju.Transposer(signal.target)

global_it = cfg['global_it']
n_samples = cfg['Nsamples']
samples = ift.optimize_kl(
    likelihood_sum,
    global_it,
    n_samples,
    minimizer,
    ic_sampling,
    None,
    export_operator_outputs={
        "signal": transpose@signal,
        "point_sources": transpose@points,
        "diffuse": transpose@diffuse,
        "power_spectrum": pspec,
    },
    output_directory="perseus_rec_2",
    initial_position=pos,
    comm=ju.library.mpi.comm,
    resume=True
)
