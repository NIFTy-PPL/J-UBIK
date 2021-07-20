import nifty7 as ift
import numpy as np
import matplotlib.pylab as plt
from lib.utils import *
from lib.output import plot_slices
from psf_likelihood import *
import mpi

ift.fft.set_nthreads(2)

npix_s = 1024      # number of spacial bins per axis
npix_e = 4
fov = 4.
elim   = (2., 10.) # energy range in keV #FIXME check limits

position_space = ift.RGSpace([npix_s, npix_s], distances=[2.*fov/npix_s])
e_space = ift.RGSpace(npix_e, distances=np.log(elim[1]/elim[0])/npix_e)
dom = ift.makeDomain([position_space, e_space])
zp_position_space = ift.RGSpace([2.*npix_s, 2. * npix_s], distances=[ 2.*fov/npix_s]) #FIXME less zeropadding enough?

info = np.load('14_6_0_observation.npy', allow_pickle= True).item()
psf_file = np.load('psf_ob0.npy', allow_pickle = True).item()
psf_arr = psf_file.val
max_idx = np.unravel_index(np.argmax(psf_arr[:,:,0]), psf_arr[:,:,0].shape)

#FIXME why are the maxima of PSF not on one spot? Statistics?
# for i in range(4):
# # THEYARE NOT EQUAL
#     print(np.argmax(psf_arr[:,:,i]))

psf_arr = np.roll(psf_arr, -max_idx[0], axis=0)
psf_arr = np.roll(psf_arr, -max_idx[1], axis=1)
psf_field = ift.Field.from_raw(dom, psf_arr)
norm = ift.ScalingOperator(dom, psf_field.integrate().val**-1)
psf_norm = norm(psf_field)


data = info['data']
exp = info['exposure']
normed_exposure = get_normed_exposure(exp, data)
normed_exposure = ift.makeOp(normed_exposure)
mask = get_mask_operator(exp)

#FIXME Need for energy correlation
# points = ift.InverseGammaOperator(zp_position_space, alpha=1.0, q=1e-4).ducktape('points')
# points_energy = ift.Simple

#FIXME put this into a config file for later runs
priors_sp_diffuse = {
        # Amplitude of field fluctuations
        'fluctuations': (0.5, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-2.5, 0.5),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (0.3, 0.05),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': None,  # 0.1, 0.5
        'prefix': 'sp_'}

priors_e_diffuse = {
        # Amplitude of field fluctuations
        'fluctuations': (0.5, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-2.5, 0.5),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (0.3, 0.05),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': None,  # 0.1, 0.5
        'prefix': 'e_'}

priors_sp_agn = {

        # Amplitude of field fluctuations
        'fluctuations': (1, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-1., 0.5),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (0.5, 0.05),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': None,  # 0.1, 0.5
        'prefix': 'sp_'}

priors_e_agn = {

        # Amplitude of field fluctuations
        'fluctuations': (1, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-1.5, 0.5),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (0.5, 0.05),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': None,  # 0.1, 0.5
        'prefix': 'e_'}

diffuse = ift.CorrelatedFieldMaker('diffuse_')
diffuse.add_fluctuations(zp_position_space, **priors_sp_diffuse)
diffuse.add_fluctuations(e_space, **priors_e_diffuse)
diffuse.set_amplitude_total_offset(offset_mean= 0, offset_std = (0.3, 0.05))
diffuse = diffuse.finalize()
diffuse = diffuse.exp()

extended = ift.CorrelatedFieldMaker('agn_')
extended.add_fluctuations(zp_position_space, **priors_sp_agn)
extended.add_fluctuations(e_space, **priors_e_agn)
extended.set_amplitude_total_offset(offset_mean= 0, offset_std = (0.3, 0.05))
extended = extended.finalize()
extended = extended.exp()

signal = diffuse + extended #+ points
signal = signal.real

zp = ift.FieldZeroPadder(dom, zp_position_space.shape, space=0, central=False)
zp_central = ift.FieldZeroPadder(dom, zp_position_space.shape, space=0, central=True)

psf = zp_central(psf_norm)
convolved = convolve_field_operator(psf, signal, space=0)
conv = zp.adjoint @ convolved
signal_response = mask @ normed_exposure @ conv

ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=3, iteration_limit=4)
ic_sampling = ift.AbsDeltaEnergyController(name='Samplig(lin)', deltaE=0.05, iteration_limit = 100)

masked_data = mask(data) #FIXME this is kind of broken
likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

minimizer_sampling = ift.NewtonCG(ift.AbsDeltaEnergyController(name="Sampling (nonlin)",
                                                               deltaE=0.5, convergence_level=2,
                                                               iteration_limit= 0))
pos = 0.1*ift.from_random(signal.domain)
if False:
    H=ift.EnergyAdapter(pos, H, want_metric=True)
    H,_ = minimizer(H)
    pos = H.position
    ift.extra.minisanity(masked_data, lambda x: ift.makeOp(1/signal_response(x)), signal_response, pos)

    dct = {'data': data_field,
           'psf_sim': psf_field,
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
        if ii >= 3:
            ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, iteration_limit=5, convergence_level=5)
            minimizer_sampling = ift.NewtonCG(ift.AbsDeltaEnergyController(name="Sampling (nonlin)",
                                                               deltaE=0.5, convergence_level=2,
                                                               iteration_limit= 10))

            minimizer = ift.NewtonCG(ic_newton)
        KL = ift.GeoMetricKL(pos, H, 4, minimizer_sampling, True, comm=mpi.comm)
        KL, _ = minimizer(KL)
        samples = list(KL.samples)
        pos= KL.position
        ift.extra.minisanity(masked_data, lambda x: ift.makeOp(1/signal_response(x)), signal_response, pos, samples)

        sc = ift.StatCalculator()
        # ps = ift.StatCalculator()
        df = ift.StatCalculator()
        sr = ift.StatCalculator()
        ex = ift.StatCalculator()
        for foo in samples:
            united = foo.unite(pos)
            sc.add(signal.force(united))
            # ps.add(points.force(united))
            df.add(diffuse.force(united))
            ex.add(extended.force(united))
            sr.add(signal_response.force(united))
        dct = {'data': data,
            'psf_sim': psf_field,
            'psf_norm': psf_norm,
            'signal_rec_mean': zp.adjoint(sc.mean),
            'signal_rec_sigma': zp.adjoint(sc.var.sqrt()),
            'diffuse': zp.adjoint(df.mean),
            'extended': zp.adjoint(ex.mean),
            # 'pointsource':zp.adjoint(ps.mean),
            'signal_response': mask.adjoint(sr.mean),
        }
        #TODO add samples
        np.save('varinf_reconstruction.npy', dct)
        np.save('lat_mean.npy', pos)
        # for oo, nn in [(extended, "extended"), (diffuse, "diffuse"), (points, "points")]:
        #     samps = []
        #     for foo in samples:
        #         samps.append(oo.force(foo.unite(KL.position)).val)
        #     np.save(f"{nn}.npy", np.array(samps))
