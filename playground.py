import nifty7 as ift
import numpy as np
import matplotlib.pylab as plt

npix_s = 256       # number of spacial bins per axis
fov = 4.
position_space = ift.RGSpace([npix_s, npix_s], distances=[ 2.*fov/npix_s])
zp_position_space = ift.RGSpace([2.*npix_s, 2. * npix_s], distances=[ 2.*fov/npix_s])

info = np.load('5_3_observation.npy', allow_pickle= True).item()
data = info['data'].val[:,:,1]
data = ift.Field.from_raw(position_space, data)


exp_field = info['exposure'].val[:,:,1]
exp_field = ift.Field.from_raw(position_space, exp_field)
plot= ift.Plot()
exp_norm = (data / exp_field).mean()
normed_exp_field =  exp_field * exp_norm.val
#FIXME MASK with 0 and NAN entrys

points = ift.InverseGammaOperator(position_space, 2.5, 1.5).ducktape('points')
args = {
        'offset_mean': 2,
        'offset_std': (1e-1, 1e-2),

        # Amplitude of field fluctuations
        'fluctuations': (0.5, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-2., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (3, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (0.2, 0.5)  # 0.1, 0.5
        }

diffuse = ift.SimpleCorrelatedField(position_space, **args).real
diffuse = ift.exp(diffuse)
# zp_signal = ift.FieldZeroPadder(position_space, zp_position_space.shape)

signal = diffuse + points
Mask = ift.DiagonalOperator(normed_exp_field)
# FFT = ift.FFTOperator(zp_position_space)

# kernel = np.zeros(zp_position_space.shape)
# kernel[0,0] = 1024
# kernel = ift.Field.from_raw(zp_position_space, kernel)
# psf = FFT(kernel)
# psf = ift.DiagonalOperator(psf)

# conv = FFT.inverse @ psf @ FFT @ diffuse
# conv = conv.real
# signal_response = Mask@ zp_signal.adjoint @ conv
signal_response = Mask@signal
ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, iteration_limit=10, convergence_level=3)
ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit = 200)
likelihood = ift.PoissonianEnergy(data) @ signal_response

minimizer = ift.NewtonCG(ic_newton)

H = ift.StandardHamiltonian(likelihood, ic_sampling)
initial_position = 0.1*ift.from_random(H.domain)
pos = initial_position


if False:
    H=ift.EnergyAdapter(pos, H, want_metric=True)
    H,_ = minimizer(H)
    plt = ift.Plot()
    plt.add(data)
    plt.add(signal_response.force(H.position))
    plt.add(diffuse.force(H.position))
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
        for foo in samples:
            sc.add((signal.force(foo.unite(KL.position))))
        plt.add(ift.log(sc.mean), title="Reconstructed Signal")
        plt.add(sc.var.sqrt(), title = "Relative Uncertainty")
        plt.add(ift.log((signal_response.force(KL.position))), title= 'signalresponse')
        plt.add(ift.log((data)), vmin= 0, title = 'data')
        plt.add((ift.abs(signal_response.force(KL.position)-data)), title = "Residual")
        plt.output(name= f'rec_{ii}.png')
