import nifty8 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import get_normed_exposure, get_mask_operator, convolve_field_operator, Transposer
from lib.output import plot_result
import lib.mpi as mpi

ift.set_nthreads(2)

npix_s = 1024  # number of spacial bins per axis
fov = 4.0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

info = np.load("chandra_4952_observation.npy", allow_pickle=True).item()
psf_file = np.load("psf_obs4952.npy", allow_pickle=True).item()

psf_arr = psf_file.val[:, :, 0]
psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
psf_field = ift.Field.from_raw(position_space, psf_arr)
norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
psf_norm = norm(psf_field)
data = info["data"].val[:, :, 0]
data_field = ift.Field.from_raw(position_space, data)

exp = info["exposure"].val[:, :, 0]
exp_field = ift.Field.from_raw(position_space, exp)
normed_exposure = get_normed_exposure(exp_field, data_field)
normed_exposure = ift.makeOp(normed_exposure)

mask = get_mask_operator(exp_field)

priors_diffuse = {
    "offset_mean": 0,
    "offset_std": (1, 0.5),
    # Amplitude of field fluctuations
    "fluctuations": (1, 1),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    "loglogavgslope": (-3.0, 1.5),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    "flexibility": (2.0, 2.0),  # 2.0, 1.0
    # How ragged the integrated Wiener process component is
    "asperity": (0.1, 0.5),  # 0.1, 0.5
    "prefix": "diffuse",
}


diffuse = ift.SimpleCorrelatedField(position_space, **priors_diffuse)
pspec = diffuse.power_spectrum
diffuse = diffuse.exp()
points = ift.InverseGammaOperator(position_space, alpha=3, q=0.7)  # 3,0.7, 1.2, 0.02
points = points.ducktape("points")
signal = points + diffuse
signal = signal.real

# p=ift.Plot()
# for i in range(10):
#     f = ift.from_random(signal.domain)
#     p.add(diffuse.force(f))
#     p.add(points.force(f))
#     p.add(signal.force(f))
# p.output(name='priorsamples.png',nx=3,ny=10, xsize=20,ysize=60, dpi=100)
# exit()
psf = psf_norm
convolved = convolve_field_operator(psf, signal)
conv = convolved

signal_response = mask @ normed_exposure @ conv

ic_newton = ift.AbsDeltaEnergyController(
    name="Newton", deltaE=0.5, iteration_limit=5, convergence_level=5
)
ic_sampling = ift.AbsDeltaEnergyController(
    name="Samplig(lin)", deltaE=0.05, iteration_limit=50
)

masked_data = mask(data_field)
likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

nl_sampling_minimizer = None
pos = 0.1 * ift.from_random(signal.domain)

if True:
    pos = ift.ResidualSampleList.load_mean("sipsf_result")
    print("loaded")


def callback(samples):
    s = ift.extra.minisanity(
        masked_data,
        lambda x: ift.makeOp(1 / signal_response(x)),
        signal_response,
        samples,
    )
    ps_mean, ps_var = samples.sample_stat(points)
    plot_result(ps_mean, "new_rec/point_sources/ps_mean.png", logscale=True, vmin=1)


global_it = 1
n_samples = 4
samples = ift.optimize_kl(
    likelihood,
    global_it,
    n_samples,
    minimizer,
    ic_sampling,
    nl_sampling_minimizer,
    plottable_operators={
        "signal": signal,
        "point_sources": points,
        "diffuse": diffuse,
        "power_spectrum": pspec,
    },
    output_directory="new_rec",
    initial_position=pos,
    comm=mpi.comm,
    callback=callback,
    overwrite=True
)

# for ii in range(10):
#     N_samples = 4
#     if ii >= 0:
#         ic_newton = ift.AbsDeltaEnergyController(
#             name="Newton", deltaE=0.5, iteration_limit=5, convergence_level=5
#         )
#         minimizer_sampling = ift.NewtonCG(
#             ift.AbsDeltaEnergyController(
#                 name="Sampling (nonlin)",
#                 deltaE=0.5,
#                 convergence_level=2,
#                 iteration_limit=10,
#             )
#         )
#         minimizer = ift.NewtonCG(ic_newton)
#         N_samples=8
#     KL = ift.SampledKLEnergy(pos, H, N_samples, minimizer_sampling, True, comm=mpi.comm)
#     KL, _ = minimizer(KL)
#     pos = KL.position
#     ift.extra.minisanity(
#         masked_data,
#         lambda x: ift.makeOp(1 / signal_response(x)),
#         signal_response,
#         KL.samples,
#     )
#     KL.samples.save("sipsf_result")
#     samples = ift.ResidualSampleList.load("sipsf_result", comm=mpi.comm)
#     mean, var = samples.sample_stat(signal)
#     ps_mean, ps_var = samples.sample_stat(points)
#     sr_mean, sr_var = samples.sample_stat(normed_exposure @ conv)
#     # plotting check

#     if mpi.master:
#         plot_result(mean, "sipsf_mean.png", logscale=False, vmin=0, vmax=10)
#         plot_result(ps_mean, "sipsf_ps_mean.png", logscale=False, vmin=0)  # ,vmax=10)
#         plot_result(var.sqrt(), "sipsf_poststd.png", logscale=False, vmin=0, vmax=3)
#         plot_result(data_field, "11713_sipsf_data.png", logscale=True, vmin=1)
#         plot_result(sr_mean, "sipsf_sr_mean.png", logscale=False, vmin=0 , vmax=10)
#         plot_result(
#             data_field - sr_mean, "sipsf_residuals", logscale=False)
