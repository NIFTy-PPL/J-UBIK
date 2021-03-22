import nifty7 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import *
from psf_likelihood import *

npix_s = 256       # number of spacial bins per axis
fov = 4.
position_space = ift.RGSpace([npix_s, npix_s], distances=[ 2.*fov/npix_s])
zp_position_space = ift.RGSpace([2.*npix_s, 2. * npix_s], distances=[ 2.*fov/npix_s])

info = np.load('5_3_observation.npy', allow_pickle= True).item()
psf_file = np.load('morecountsobservation.npy', allow_pickle = True).item()

psf_arr = psf_file['psf_sim'].val[:, : ,1]
psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
psf_field = ift.Field.from_raw(position_space, psf_arr)

psf_likelihood, psf_model = makePSFmodel(psf_field)
norm = ift.ScalingOperator(position_space, psf_field.integrate().val**-1)

psf_model = norm @ psf_model
#TODO THIS IS NOT CORRECT
#TODO Normalize PSF

data = info['data'].val[:, :, 1]
data_field = ift.Field.from_raw(position_space, data)

exp = info['exposure'].val[:, :, 1]
exp_field = ift.Field.from_raw(position_space, exp)
normed_exposure = get_normed_exposure_operator(exp_field, data)

mask = get_mask_operator(exp_field)

points = ift.InverseGammaOperator(zp_position_space, alpha=0.7, q=1e-4).ducktape('points')
#TODO FIXME this prior is broken...

priors_diffuse = {'offset_mean': 4,
        'offset_std': (2, .1),

        # Amplitude of field fluctuations
        'fluctuations': (1.5, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-1.5, 0.5),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1.5, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (0.2, 0.5),  # 0.1, 0.5
        'prefix': 'diffuse'}
diffuse = ift.SimpleCorrelatedField(zp_position_space, **priors_diffuse)
diffuse = diffuse.exp()

signal = diffuse + points
signal = signal.real

zp = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=False)
#signal = zp @ signal

zp_central = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=True)
psf = zp_central(psf_model)

convolved = convolve_operators(psf, signal)
conv = zp.adjoint @ convolved

signal_response = mask @ normed_exposure @ conv

ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, iteration_limit=10, convergence_level=3)
ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit = 50)
masked_data = mask(data_field)

psf_pos = minimizePSF(psf_likelihood, iterations=10)


likelihood = ift.PoissonianEnergy(masked_data) @ signal_response

minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

signal_pos = 0.1*ift.from_random(signal.domain)


pos = signal_pos.unite(psf_pos)

if False:
    H=ift.EnergyAdapter(pos, H, want_metric=True, constants=psf_pos.keys())
    H,_ = minimizer(H)
    pos = H.position.unite(psf_pos)
    plt = ift.Plot()
    plt.add(ift.log10(psf_field))
    plt.add(ift.log10(psf_model.force(pos)))
    plt.add(ift.log10(zp.adjoint(signal.force(pos))), title = 'signal_rec')
    plt.add(ift.log10(points.force(pos)),vmin=0, title ="stars")
    plt.add(ift.log10(diffuse.force(pos)), title="diffuse")
    plt.add(ift.log10(data_field), title='data')
    plt.add(ift.log10(mask.adjoint(signal_response.force(pos))), title='signal_response')
    plt.add((ift.abs(mask.adjoint(signal_response.force(pos))-data_field)), title = "Residual")
    plt.output(ny =2 , nx = 4, xsize= 30, ysize= 15, name='map.png')
else:
    for ii in range(10):
        KL = ift.MetricGaussianKL.make(pos, H, 5, True, constants= psf_pos.keys())
        KL, _ = minimizer(KL)
        pos = KL.position.unite(psf_pos)
        samples = list(KL.samples)
        ift.extra.minisanity(masked_data, lambda x: ift.makeOp(1/signal_response(x)), signal_response, pos, samples)

        plt = ift.Plot()
        sc = ift.StatCalculator()
        ps = ift.StatCalculator()
        df = ift.StatCalculator()
        for foo in samples:
            united = foo.unite(pos)
            sc.add(signal.force(united))
            ps.add(points.force(united))
            df.add(diffuse.force(united))
        plt.add(ift.log10(ps.mean), title="PointSources")
        plt.add(ift.log10(df.mean), title="diffuse")
        plt.add(ift.log10(zp.adjoint(sc.mean)), title="Reconstructed Signal")
        plt.add(zp.adjoint(sc.var.sqrt()), title = "Relative Uncertainty")
        plt.add(ift.log10((mask.adjoint(signal_response.force(pos)))), title= 'signalresponse')
        plt.add(ift.log10(data_field), vmin= 0, title = 'data')
        plt.add((ift.abs(mask.adjoint(signal_response.force(pos))-data_field)), title = "Residual")
        plt.output(ny=2, nx=4, xsize=100, ysize= 40,name= f'rec_{ii}.png')
