import nifty8 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import get_normed_exposure, get_mask_operator, convolve_field_operator
import lib.mpi as mpi

ift.set_nthreads(2)

npix_s = 1024  # number of spacial bins per axis
fov = 4.0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])
zp_position_space = ift.RGSpace(
    [2.0 * npix_s, 2.0 * npix_s], distances=[2.0 * fov / npix_s]
)

info = np.load("5_10_0_observation.npy", allow_pickle=True).item()
psf_file = np.load("psf_ob0.npy", allow_pickle=True).item()

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

points = ift.InverseGammaOperator(zp_position_space, alpha=1.0, q=1e-4).ducktape(
    "points"
)

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

diffuse = ift.SimpleCorrelatedField(zp_position_space, **priors_diffuse)
diffuse = diffuse.exp()
extended = ift.SimpleCorrelatedField(zp_position_space, **priors_extended_points)
extended = extended.exp()

signal = diffuse + extended + points
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
    name="Samplig(lin)", deltaE=0.05, iteration_limit=100
)

masked_data = mask(data_field)
likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

minimizer_sampling = ift.NewtonCG(
    ift.AbsDeltaEnergyController(
        name="Sampling (nonlin)", deltaE=0.5, convergence_level=2, iteration_limit=0
    )
)
pos = 0.1 * ift.from_random(signal.domain)
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
        if ii >= 3:
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
        KL = ift.GeoMetricKL(pos, H, 4, minimizer_sampling, True, comm=mpi.comm)
        KL, _ = minimizer(KL)
        samples = list(KL.samples)
        pos = KL.position
        ift.extra.minisanity(
            masked_data,
            lambda x: ift.makeOp(1 / signal_response(x)),
            signal_response,
            pos,
            samples,
        )

        sc = ift.StatCalculator()
        ps = ift.StatCalculator()
        df = ift.StatCalculator()
        sr = ift.StatCalculator()
        ex = ift.StatCalculator()
        for foo in samples:
            united = foo.unite(pos)
            sc.add(signal.force(united))
            ps.add(points.force(united))
            df.add(diffuse.force(united))
            ex.add(extended.force(united))
            sr.add(signal_response.force(united))
        dct = {
            "data": data_field,
            "psf_sim": psf_field,
            "psf_norm": psf_norm,
            "signal_rec_mean": zp.adjoint(sc.mean),
            "signal_rec_sigma": zp.adjoint(sc.var.sqrt()),
            "diffuse": zp.adjoint(df.mean),
            "extended": zp.adjoint(ex.mean),
            "pointsource": zp.adjoint(ps.mean),
            "signal_response": mask.adjoint(sr.mean),
        }
        # TODO add samples
        np.save("varinf_reconstruction.npy", dct)

        for oo, nn in [
            (extended, "extended"),
            (diffuse, "diffuse"),
            (points, "points"),
        ]:
            samps = []
            for foo in samples:
                samps.append(oo.force(foo.unite(KL.position)).val)
            np.save(f"{nn}.npy", np.array(samps))
