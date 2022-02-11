import nifty8 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import get_normed_exposure, get_mask_operator, convolve_field_operator, Transposer
from lib.output import plot_result
import lib.mpi as mpi
import yaml

with open("config.yaml", 'r') as cfg_file:
    cfg = yaml.safe_load(cfg_file)

#TODO Write "Build Model" and Build Likelihood

npix_s = 1024  # number of spacial bins per axis
fov = 4.0
energy_bin = 0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

info = np.load(cfg['dataset']['observation'], allow_pickle=True).item()
psf_file = np.load("psf_obs4952.npy", allow_pickle=True).item()

### PSF1
psf_arr = psf_file.val[:, :, energy_bin]
psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
psf_field = ift.Field.from_raw(position_space, psf_arr)
norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
psf_norm = norm(psf_field)

### PSF2
psf_arr = psf_file.val[:, :, energy_bin]
psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
psf_field = ift.Field.from_raw(position_space, psf_arr)
norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
psf_norm = norm(psf_field)

#Data 1
data = info["data"].val[:, :, energy_bin]
data_field = ift.Field.from_raw(position_space, data)


#Data 2
data = info["data"].val[:, :, energy_bin]
data_field = ift.Field.from_raw(position_space, data)

#Exp 1
exp = info["exposure"].val[:, :, energy_bin]
exp_field = ift.Field.from_raw(position_space, exp)
# normed_exposure = get_normed_exposure(exp_field, data_field)
normed_exposure = ift.makeOp(exp_field)

#Exp 1
exp = info["exposure"].val[:, :, energy_bin]
exp_field = ift.Field.from_raw(position_space, exp)
# normed_exposure = get_normed_exposure(exp_field, data_field)
normed_exposure = ift.makeOp(exp_field)

#Mask 1
mask = get_mask_operator(exp_field)

#Mask 2
mask = get_mask_operator(exp_field)

#Model
diffuse = ift.SimpleCorrelatedField(position_space, **cfg['priors_diffuse'])
pspec = diffuse.power_spectrum
diffuse = diffuse.exp()
points = ift.InverseGammaOperator(position_space, **cfg['points'])
points = points.ducktape("points")
signal = points + diffuse
signal = signal.real

#Likelihood 1
transpose = Transposer(signal.target)
psf = psf_norm
convolved = convolve_field_operator(psf, signal)
conv = convolved
signal_response = mask @ normed_exposure @ conv


#Likelihood 2
transpose = Transposer(signal.target)
psf = psf_norm
convolved = convolve_field_operator(psf, signal)
conv = convolved
signal_response = mask @ normed_exposure @ conv


ic_newton = ift.AbsDeltaEnergyController(**cfg['ic_newton'])
ic_sampling = ift.AbsDeltaEnergyController(**cfg['ic_sampling'])

masked_data = mask(data_field)
likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

nl_sampling_minimizer = None
pos = 0.1 * ift.from_random(signal.domain)


def callback(samples):
    s = ift.extra.minisanity(
        masked_data,
        lambda x: ift.makeOp(1 / signal_response(x)),
        signal_response,
        samples,
    )
    ps_mean, ps_var = samples.sample_stat(points)
    sr_mean, sr_var = samples.sample_stat(mask.adjoint(signal_response))
    plot_result(ps_mean, "new_rec/point_sources/ps_mean.png", logscale=True, vmin=1)
    plot_result(sr_mean, "new_rec/sr_mean.png", logscale=True, vmin=1)
    plot_result(mask.adjoint(masked_data)-sr_mean, "new_rec/residuals.png", logscale=False)


global_it = 1
n_samples = 8
samples = ift.optimize_kl(
    likelihood,
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
    output_directory="new_rec",
    initial_position=pos,
    comm=mpi.comm,
    inspect_callback=callback,
    overwrite=True
)

