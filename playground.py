import nifty7 as ift
import numpy as np
import matplotlib.pylab as plt

npix_s = 256       # number of spacial bins per axis
fov = 4.
position_space = ift.RGSpace([npix_s, npix_s], distances=[ 2.*fov/npix_s])
info = np.load('262_observation.npy', allow_pickle= True).item()
data = info['data'].val[:,:,1]
data = ift.Field.from_raw(position_space, data)


exp_field = info['exposure'].val[:,:,1]
exp_field = ift.Field.from_raw(position_space, exp_field)

exp_norm = (data / exp_field).mean()

normed_exp_field =  exp_field * exp_norm.val

args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (2., 1.),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-4., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (5, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.5)  # 0.1, 0.5
        }

diffuse = ift.SimpleCorrelatedField(position_space, **args).exp()
signal = diffuse #+ point_sources
Mask = ift.DiagonalOperator(normed_exp_field)
if True:
        signal_response = Mask(signal)
else:
    signal_response = signal
ic_newton = ift.DeltaEnergyController(
name='Newton', iteration_limit=50, tol_rel_deltaE=1e-8)
minimizer = ift.NewtonCG(ic_newton)

likelihood = ift.PoissonianEnergy(data) @ signal_response
H = ift.StandardHamiltonian(likelihood)


initial_position = ift.from_random(H.domain)
H = ift.EnergyAdapter(initial_position, H, want_metric=True)
H, _ = minimizer(H)

plt = ift.Plot()
plt.add((data), vmin= 0, title = 'data')
plt.add((signal.force(H.position)), vmin=0, title = 'signal')
plt.add((signal_response.force(H.position)), vmin=0, title= 'signalresponse')
plt.output()
