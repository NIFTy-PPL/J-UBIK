import nifty7 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import *
from psf_likelihood import *
import mpi

ift.fft.set_nthreads(8)

npix_s = 1024      # number of spacial bins per axis
fov = 4.
position_space = ift.RGSpace([npix_s, npix_s], distances=[ 2.*fov/npix_s])
zp_position_space = ift.RGSpace([2.*npix_s, 2. * npix_s], distances=[ 2.*fov/npix_s])

info = np.load('14_6_0_observation.npy', allow_pickle= True).item()
psf_file = np.load('psf_ob0.npy', allow_pickle = True).item()

psf_arr = psf_file.val[:, : ,0]
psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
psf_field = ift.Field.from_raw(position_space, psf_arr)

psf_likelihood, psf_model = makePSFmodel(psf_field)
psf_pos = minimizePSF(psf_likelihood, iterations=20)
norm = ift.ScalingOperator(position_space, psf_field.integrate().val**-1)

ift.extra.minisanity(psf_field, lambda x: ift.makeOp(1/psf_model(x)), psf_model, psf_pos)
psf_model = norm @ psf_model

#TODO Normalize PSF (at the right moment)
#FIXME AGN not the same for diferent obs

data = info['data'].val[:, :, 0]
data_field = ift.Field.from_raw(position_space, data)

exp = info['exposure'].val[:, :, 0]
exp_field = ift.Field.from_raw(position_space, exp)
normed_exposure = get_normed_exposure_operator(exp_field, data)

mask = get_mask_operator(exp_field)

points = ift.InverseGammaOperator(zp_position_space, alpha=1.0, q=1e-4).ducktape('points')
priors_diffuse = {'offset_mean': 0,
        'offset_std': (2, .1),

        # Amplitude of field fluctuations
        'fluctuations': (1.9, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-2.0, 1.0),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2.0, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (1, 0.5),  # 0.1, 0.5
        'prefix': 'diffuse'}

diffuse = ift.SimpleCorrelatedField(zp_position_space, **priors_diffuse)
diffuse = diffuse.exp()
signal = diffuse +points#+ clusters
signal = signal.real

zp = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=False)
zp_central = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=True)

psf = zp_central(psf_model)
convolved = convolve_operators(psf, signal)
conv = zp.adjoint @ convolved

signal_response = mask @ normed_exposure @ conv

ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, iteration_limit=3, convergence_level=3)
ic_sampling = ift.AbsDeltaEnergyController(name='Samplig(lin)',deltaE=0.05, iteration_limit = 30)

masked_data = mask(data_field)
likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

signal_pos = 0.1*ift.from_random(signal.domain)
minimizer_sampling = ift.NewtonCG(ift.AbsDeltaEnergyController(name="Sampling (nonlin)",
                                                               deltaE=0.5, convergence_level=2,
                                                               iteration_limit= 10))
pos = signal_pos.unite(psf_pos)

if False:
    H=ift.EnergyAdapter(pos, H, want_metric=True, constants=psf_pos.keys())
    H,_ = minimizer(H)
    pos = H.position.unite(psf_pos)
    ift.extra.minisanity(masked_data, lambda x: ift.makeOp(1/signal_response(x)), signal_response, pos)

    dct = {'data': data_field,
           'psf_sim': psf_field,
           'psf_fit': psf_model.force(pos),
           'signal_rec': zp.adjoint(signal.force(pos)),
           'signal_conv': conv.force(pos),
           'diffuse': zp.adjoint(diffuse.force(pos)),
           'pointsource':zp.adjoint(points.force(pos)),
           'signal_response': mask.adjoint(signal_response.force(pos)),
           'residual': ift.abs(mask.adjoint(signal_response.force(pos))-data_field),
    }
    np.save('map_reconstruction.npy', dct)
else:
    for ii in range(10):
        KL = ift.GeoMetricKL(pos, H, 2, minimizer_sampling, True, constants= psf_pos.keys(), point_estimates= psf_pos.keys(), comm=mpi.comm)
        KL, _ = minimizer(KL)
        pos = KL.position.unite(psf_pos)
        samples = list(KL.samples)
        ift.extra.minisanity(masked_data, lambda x: ift.makeOp(1/signal_response(x)), signal_response, pos, samples)

        plt = ift.Plot()
        sc = ift.StatCalculator()
        ps = ift.StatCalculator()
        df = ift.StatCalculator()
        sr = ift.StatCalculator()
        for foo in samples:
            united = foo.unite(pos)
            sc.add(signal.force(united))
            ps.add(points.force(united))
            df.add(diffuse.force(united))
            sr.add(signal_response.force(united))
        dct = {'data': data_field,
            'psf_sim': psf_field,
            'psf_fit': psf_model.force(pos),
            'signal_rec_mean': zp.adjoint(sc.mean),
            'signal_rec_sigma': zp.adjoint(sc.var.sqrt()),
            'diffuse': zp.adjoint(df.mean),
            'pointsource':zp.adjoint(ps.mean),
            'signal_response': mask.adjoint(sr.mean),
        }
        np.save('varinf_reconstruction.npy', dct)
