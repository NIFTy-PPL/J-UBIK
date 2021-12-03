import nifty8 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import get_normed_exposure, get_mask_operator, convolve_field_operator
from lib.output import plot_result
import lib.mpi as mpi

ift.set_nthreads(2)

npix_s = 1024  # number of spacial bins per axis
fov = 4.0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

info = np.load("5_10_0_observation.npy", allow_pickle=True).item()
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
    "offset_std": (0.3, 0.05),
    # Amplitude of field fluctuations
    "fluctuations": (0.5, 0.5),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    "loglogavgslope": (-2.5, 0.5),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    "flexibility": (0.3, 0.05),  # 2.0, 1.0
    # How ragged the integrated Wiener process component is
    "asperity": None,  # 0.1, 0.5
    "prefix": "diffuse",
}

priors_extended_points = {
    "offset_mean": 0,
    "offset_std": (0.3, 0.05),
    # Amplitude of field fluctuations
    "fluctuations": (2, 0.5),  # 1.0, 1e-2
    # Exponent of power law power spectrum component
    "loglogavgslope": (-0.5, 0.5),  # -6.0, 1
    # Amplitude of integrated Wiener process power spectrum component
    "flexibility": (1, 0.05),  # 2.0, 1.0
    # How ragged the integrated Wiener process component is
    "asperity": None,  # 0.1, 0.5
    "prefix": "extended",
}

diffuse = ift.SimpleCorrelatedField(position_space, **priors_diffuse)
diffuse = diffuse.exp()

## Other Components
# extended = ift.SimpleCorrelatedField(zp_position_space, **priors_extended_points)
# extended = extended.exp()
# points = ift.InverseGammaOperator(zp_position_space, alpha=1.0, q=1e-4).ducktape(
#     "points"
# )

signal = diffuse  # + extended + points
signal = signal.real
zp = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=False)
zp_central = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=True)

psf = zp_central(psf_norm)
convolved = convolve_field_operator(psf, signal)
conv = zp.adjoint @ convolved
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

minimizer_sampling = None
pos = 0.1 * ift.from_random(signal.domain)

if True:
    pos = ift.ResidualSampleList.load_mean("sipsf_result")
    print('loaded')

    ift.extra.minisanity(
        masked_data, lambda x: ift.makeOp(1 / signal_response(x)), signal_response, pos
    )
    for ii in range(1):
    #     if ii >= 1:
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
    #     KL = ift.SampledKLEnergy(pos, H, 4, minimizer_sampling, True, comm=mpi.comm)
    #     KL, _ = minimizer(KL)
    #     pos = KL.position
    #     ift.extra.minisanity(
    #         masked_data,
    #         lambda x: ift.makeOp(1 / signal_response(x)),
    #         signal_response,
    #         KL.samples,
    #     )
    #     KL.samples.save("result")
        samples = ift.ResidualSampleList.load("result")#, mpi.comm)
        mean, var = samples.sample_stat(zp.adjoint(signal))
        # plotting check

        if mpi.master:
            plot_result(mean, "test_mean.png", logscale=False, vmin=0,vmax=10)
            plot_result(var.sqrt(), "test_poststd.png", vmin=0, vmax=3)
            plot_result(data_field, "data.png", vmin=0, vmax=10)
        plot_result(mean, "sipsf_mean.png", logscale=False, vmin=0,vmax=10)
        plot_result(ps_mean, "sipsf_ps_mean.png", logscale=False, vmin=0)#,vmax=10)
        plot_result(var.sqrt(), "sipsf_poststd.png",logscale=False, vmin=0, vmax=3)
        plot_result(data_field, "11713_sipsf_data.png", logscale=False, vmin=0, vmax=10)
        plot_result(sr_mean, "sipsf_sr_mean.png", logscale=True, vmin=1)#, vmax=10)
        plot_result(ift.abs(data_field-sr_mean), "sipsf_residuals", logscale=True,vmin=1)
