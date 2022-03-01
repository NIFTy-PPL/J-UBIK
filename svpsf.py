import numpy as np
import matplotlib.pylab as plt

import nifty8 as ift
import xubik0 as xu

ift.set_nthreads(2)

# FIXME Discuss normalization
# TODO Think about shapes and if this is really the right thing now
# FIXME THIS SHOULD BE THE MARGIN ZEROPADDER

npix_s = 1024
fov = 4.0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

info = np.load("chandra_11713_observation.npy", allow_pickle=True).item()
psf_file = np.load("patches_psf_rolled.npy", allow_pickle=True).item()

psf_file = psf_file["psf_sim"]
psfs = []
for p in psf_file:
    psfs.append(p.val)
psfs = np.array(psfs, dtype="float64")

data = info["data"].val[:, :, 0]
data_field = ift.Field.from_raw(position_space, data)

exp = info["exposure"].val[:, :, 0]
exp_field = ift.Field.from_raw(position_space, exp)
normed_exposure = xu.get_normed_exposure(exp_field, data_field)
normed_exposure = ift.makeOp(normed_exposure)
mask = xu.get_mask_operator(exp_field)


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

diffuse = ift.SimpleCorrelatedField(position_space, **priors_diffuse).exp()

conv_op = OverlapAddConvolver(signal.target, psfs, 64, 16)
conv = conv_op @ signal
cut = ift.FieldZeroPadder(signal.target, conv.target.shape).adjoint
cut = xu.MarginZeroPadder(signal.target, ((conv.target.shape[0] -signal.target.shape[0])//2), space=0).adjoint
conv = cut @ conv

def coord_center(side_length, side_n):
    tdx = tdy = side_length // side_n
    xc = np.arange(tdx // 2, tdx * side_n, tdx // 2)
    yc = np.arange(tdy // 2, tdy * side_n, tdy // 2)
    co = np.array(np.meshgrid(xc, yc)).reshape(2, -1)
    # res = np.ravel_multi_index(co, [side_length, side_length])
    return co


coords = coord_center(1024, 8)
z = np.zeros([1024, 1024])
for a in coords[0]:
    for b in coords[1]:
        z[a, b] = 10000
zf = ift.makeField(position_space, z)

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
    pos = ift.ResidualSampleList.load_mean("result_new")
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
    for ii in range(10):
        if ii >= 0:
            ic_newton = ift.AbsDeltaEnergyController(
                name="Newton", deltaE=0.5, iteration_limit=5, convergence_level=5
            )
            minimizer_sampling = ift.NewtonCG(
                ift.AbsDeltaEnergyController(
                    name="Sampling (nonlin)",
                    deltaE=0.5,
                    convergence_level=2,
                    iteration_limit=10,
                )
            )

            minimizer = ift.NewtonCG(ic_newton)
        KL = ift.SampledKLEnergy(pos, H, 4, minimizer_sampling, True, comm=xu.library.mpi.comm)
        KL, _ = minimizer(KL)
        pos = KL.position
        ift.extra.minisanity(
            masked_data,
            lambda x: ift.makeOp(1 / signal_response(x)),
            signal_response,
            KL.samples,
        )
        KL.samples.save("result_new")
        samples = ift.ResidualSampleList.load("result_new", comm=mpi.comm)
        mean, var = samples.sample_stat(signal)
        sr_mean, sr_var = samples.sample_stat(normed_exposure @ conv)
        # plotting check

        if mpi.master:
            xu.plot_result(mean, "svpsf_mean.png", logscale=False, vmin=0,vmax=10)
            xu.plot_result(sr_mean, "svpsf_sr_mean.png", logscale=False, vmin=0,vmax=10)
            xu.plot_result(var.sqrt(), "svpsf_poststd.png", logscale=False , vmin=0, vmax=3)
            xu-plot_result(data_field, "11713_data.png", logscale=False, vmin=0, vmax=10)
