import nifty8 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import get_normed_exposure, get_mask_operator, convolve_field_operator
from lib.output import plot_result
import lib.mpi as mpi
from lib.PatchingOperator import OverlappAdd
from lib.PaddingOperators import MarginZeroPadder
from lib.InterpolationWeights import get_weights

ift.set_nthreads(2)


def OverlapAddConvolver(domain, kernels_arr, n, margin):
    oa = OverlappAdd(domain[0], n, 0)
    weights = ift.makeOp(get_weights(oa.target))
    zp = MarginZeroPadder(oa.target, margin, space=1)
    padded = zp @ weights @ oa
    cutter = ift.FieldZeroPadder(padded.target, kernels_arr.shape[1:], space=1).adjoint
    kernels_b = ift.Field.from_raw(cutter.domain, kernels_arr)
    kernels = cutter(kernels_b)
    convolved = convolve_field_operator(kernels, padded, space=1)
    pad_space = ift.RGSpace(
        [domain.shape[0] + 2 * margin, domain.shape[1] + 2 * margin],
        distances=domain[0].distances,
    )
    oa_back = OverlappAdd(pad_space, n, margin)
    res = oa_back.adjoint @ convolved
    return res


npix_s = 1024  # number of spacial bins per axis
fov = 4.0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

info = np.load("chandra_11713_observation.npy", allow_pickle=True).item()
psf_file = np.load("patches_psf_rolled.npy", allow_pickle=True).item()

# FIXME Discuss normalization
psf_file = psf_file["psf_sim"]
psfs = []
for p in psf_file:
    psfs.append(p.val)
psfs = np.array(psfs, dtype="float64")

# psf_arr = psf_file.val[:, :, 0]
# psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
# psf_field = ift.Field.from_raw(position_space, psf_arr)
# norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
# psf_norm = norm(psf_field)

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

diffuse = ift.SimpleCorrelatedField(position_space, **priors_diffuse)
diffuse = diffuse.exp()

signal = diffuse
conv_op = OverlapAddConvolver(signal.target, psfs, 64, 64)
# TODO Think about shapes and if this is really the right thing now
conv = conv_op @ signal
cut = ift.FieldZeroPadder(signal.target, conv.target.shape).adjoint
conv = cut @ conv

p= ift.Plot()
p.add(conv(ift.from_random(conv.domain)))
p.output()
exit()
exit()
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

if False:
    pos = ift.ResidualSampleList.load_mean("result")

if False:
    H = ift.EnergyAdapter(pos, H, want_metric=True)
    H, _ = minimizer(H)
    pos = H.position
    ift.extra.minisanity(
        masked_data, lambda x: ift.makeOp(1 / signal_response(x)), signal_response, pos
    )

    dct = {
        "data": data_field,
        "psf_sim": psf_field,
        "signal_rec": zp.adjoint(signal.force(pos)),
        "signal_conv": conv.force(pos),
        "diffuse": zp.adjoint(diffuse.force(pos)),
        "pointsource": zp.adjoint(points.force(pos)),
        "signal_response": mask.adjoint(signal_response.force(pos)),
        "residual": ift.abs(mask.adjoint(signal_response.force(pos)) - data_field),
    }
    np.save("map_reconstruction.npy", dct)

else:
    for ii in range(1):
        # if ii >= 1:
        #     ic_newton = ift.AbsDeltaEnergyController(
        #         name="Newton", deltaE=0.5, iteration_limit=5, convergence_level=5
        #     )
        #     minimizer_sampling = ift.NewtonCG(
        #         ift.AbsDeltaEnergyController(
        #             name="Sampling (nonlin)",
        #             deltaE=0.5,
        #             convergence_level=2,
        #             iteration_limit=10,
        #         )
        #     )

        #     minimizer = ift.NewtonCG(ic_newton)
        # KL = ift.SampledKLEnergy(pos, H, 4, minimizer_sampling, True, comm=mpi.comm)
        # KL, _ = minimizer(KL)
        # pos = KL.position
        # ift.extra.minisanity(
        #     masked_data,
        #     lambda x: ift.makeOp(1 / signal_response(x)),
        #     signal_response,
        #     KL.samples,
        # )
        # KL.samples.save("result_new")
        samples = ift.ResidualSampleList.load("result_new", comm=mpi.comm)
        mean, var = samples.sample_stat(signal)
        sr_mean, sr_var = samples.sample_stat(conv)
        # plotting check

        if mpi.master:
            plot_result(mean, "svpsf_mean.png", logscale=True)  # vmin=0,vmax=10)
            plot_result(sr_mean, "svpsf_sr_mean.png", logscale=True)  # vmin=0,vmax=10)
            plot_result(var.sqrt(), "svpsf_poststd.png", logscale=True)  # , vmin=0, vmax=3)
            plot_result(data_field, "11713_data.png", logscale=True)  # , logscalevmin=0, vmax=10)
