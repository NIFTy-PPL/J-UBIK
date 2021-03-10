import nifty7 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import *

npix_s = 256       # number of spacial bins per axis
fov = 4.
position_space = ift.RGSpace([npix_s, npix_s], distances=[ 2.*fov/npix_s])
zp_position_space = ift.RGSpace([2.*npix_s, 2. * npix_s], distances=[ 2.*fov/npix_s])

info = np.load('5_3_observation.npy', allow_pickle= True).item()
data = info['data'].val[:, :, 1]
data_field = ift.Field.from_raw(position_space, data)

exp = info['exposure'].val[:, :, 1]
exp_field = ift.Field.from_raw(position_space, exp)
normed_exposure = get_normed_exposure_operator(exp_field, data)

mask = get_mask_operator(exp_field)

points = ift.InverseGammaOperator(position_space, 1.8, 0.5).ducktape('points')
star = ift.ValueInserter(position_space, [90, 104]).ducktape('cstar')
star = star * 3
star = star.exp()
priors_diffuse = {'offset_mean': 4,
        'offset_std': (2, .1),

        # Amplitude of field fluctuations
        'fluctuations': (1.5, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1.5, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (0.2, 0.5),  # 0.1, 0.5
        'prefix': 'diffuse'}
diffuse = ift.SimpleCorrelatedField(position_space, **priors_diffuse)
diffuse = diffuse.exp()

signal = star #diffuse + points + star
signal = signal.real

zp = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=False)
signal = zp @ signal



priors_psf = {
        'offset_mean': 1.,
        'offset_std': (1e-2, 1e-4),

        # Amplitude of field fluctuations
        'fluctuations': (1.0, 1e-2),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-4., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1., .01),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': None,  # 0.1, 0.5
        'prefix': 'psf'}
psf = ift.SimpleCorrelatedField(zp_position_space, **priors_psf)
psf = psf.exp()
convolved = convolve_operators(psf, signal)
conv = zp.adjoint @ convolved

signal_response = mask @ normed_exposure @ zp.adjoint@ signal

ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, iteration_limit=50, convergence_level=3)
ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit = 200)
masked_data = mask(data_field)
likelihood = ift.PoissonianEnergy(masked_data) @ signal_response

minimizer = ift.NewtonCG(ic_newton)

H = ift.StandardHamiltonian(likelihood, ic_sampling)
initial_position = 0.1*ift.from_random(H.domain)
pos = initial_position


if True:
    H=ift.EnergyAdapter(pos, H, want_metric=True)
    H,_ = minimizer(H)
    plt = ift.Plot()
    plt.add(ift.log10(data_field))
    plt.add(ift.log10(mask.adjoint(signal_response.force(H.position))))
    plt.add(ift.log10(zp.adjoint(signal.force(H.position))))
    # plt.add(ift.log10(zp.adjoint(psf.force(H.position))))
    plt.add(ift.log10(star.force(H.position)))
    plt.add((ift.abs(mask.adjoint(signal_response.force(H.position))-data_field)), title = "Residual")

    plt.output()
else:
    for ii in range(10):
        KL = ift.MetricGaussianKL.make(pos, H, 5, True)
        KL, _ = minimizer(KL)
        pos = KL.position
        samples = list(KL.samples)
        ift.extra.minisanity(data, lambda x: ift.makeOp(signal_response(x)), signal_response, pos, samples)

        plt = ift.Plot()
        sc = ift.StatCalculator()
        ps = ift.StatCalculator()
        df = ift.StatCalculator()
        for foo in samples:
            sc.add(signal.force(foo.unite(KL.position)))
            ps.add(points.force(foo.unite(KL.position)))
            df.add(diffuse.force(foo.unite(KL.position)))
        plt.add(ift.log10(ps.mean), title="PointSources")
        plt.add(ift.log10(df.mean), title="diffuse")
        plt.add(ift.log10(sc.mean), title="Reconstructed Signal")
        plt.add(sc.var.sqrt(), title = "Relative Uncertainty")
        plt.add(ift.log10((mask.adjoint(signal_response.force(KL.position)))), title= 'signalresponse')
        plt.add(ift.log10(data_field), vmin= 0, title = 'data')
        plt.add((ift.abs(mask.adjoint(signal_response.force(KL.position))-data_field)), title = "Residual")
        plt.output(name= f'rec_{ii}.png')
