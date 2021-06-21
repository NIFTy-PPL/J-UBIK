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

psf_arr = psf_file.val[:, : ,1]
psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
psf_field = ift.Field.from_raw(position_space, psf_arr)

psf_likelihood, psf_model = makePSFmodel(psf_field)
psf_pos = minimizePSF(psf_likelihood, iterations=10)
norm = ift.ScalingOperator(position_space, psf_field.integrate().val**-1)

ift.extra.minisanity(psf_field, lambda x: ift.makeOp(1/psf_model(x)), psf_model, psf_pos)
psf_model = norm @ psf_model

# p = ift.Plot()
# # p.add(ift.log10(psf_model.force(psf_pos)))
# # p.add(ift.log10(norm(psf_model.force(psf_pos))))
# # p.output(dpi =300)
#TODO THIS IS NOT CORRECT
#TODO Normalize PSF
#FIXME Starposition not the same for diferent obs
data = info['data'].val[:, :, 1]
data_field = ift.Field.from_raw(position_space, data)

exp = info['exposure'].val[:, :, 1]
exp_field = ift.Field.from_raw(position_space, exp)
normed_exposure = get_normed_exposure_operator(exp_field, data)

mask = get_mask_operator(exp_field)
cluster_center = ift.ValueInserter(zp_position_space, (360, 419))
cluster_val = ift.ScalingOperator(cluster_center.domain, 1).exp()
cluster_min = ift.Adder(1)
cluster_val = cluster_min @ cluster_val
cluster = cluster_center@ cluster_val
cluster = cluster.ducktape('center')

points = ift.InverseGammaOperator(zp_position_space, alpha=1.0, q=1e-4).ducktape('points')
#TODO FIXME this prior is broken...
# prior_sample_plotter(points, 5)
priors_diffuse = {'offset_mean': 0,
        'offset_std': (2, .1),

        # Amplitude of field fluctuations
        'fluctuations': (1.9, 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3.0, 1.0),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2.0, 2.),  # 2.0, 1.0

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.1),  # 0.1, 0.5
        'prefix': 'diffuse'}
diffuse = ift.SimpleCorrelatedField(zp_position_space, **priors_diffuse)
diffuse = diffuse.exp()

signal = diffuse + cluster #+points
signal = signal.real

zp = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=False)
#signal = zp @ signal

zp_central = ift.FieldZeroPadder(position_space, zp_position_space.shape, central=True)
psf = zp_central(psf_model)

convolved = convolve_operators(psf, signal)
conv = zp.adjoint @ convolved

signal_response = mask @ normed_exposure @ conv

ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, iteration_limit=30, convergence_level=3)
ic_sampling = ift.AbsDeltaEnergyController(name='Samplig(lin)',deltaE=0.05, iteration_limit = 50)
masked_data = mask(data_field)


likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
ift.exec_time(likelihood)
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

signal_pos = 0.1*ift.from_random(signal.domain)

minimizer_sampling = ift.NewtonCG(ift.AbsDeltaEnergyController(name="Sampling (nonlin)",
                                                               deltaE=0.5, convergence_level=2,
                                                               iteration_limit= 10))
pos = signal_pos.unite(psf_pos)

if True:
    H=ift.EnergyAdapter(pos, H, want_metric=True, constants=psf_pos.keys())
    H,_ = minimizer(H)
    pos = H.position.unite(psf_pos)
    ift.extra.minisanity(masked_data, lambda x: ift.makeOp(1/signal_response(x)), signal_response, pos)
    plt = ift.Plot()
    plt.add(ift.log10(psf_field))
    plt.add(ift.log10(psf_model.force(pos)))
    plt.add(ift.log10(zp.adjoint(signal.force(pos))), title = 'signal_rec', cmap ='inferno')
    # plt.add(ift.log10(points.force(pos)),vmin=0, title ="stars")
    plt.add((cluster.force(pos)),vmin=0, title ="center")
    plt.add(ift.log10(diffuse.force(pos)), title="diffuse")
    plt.add(ift.log10(data_field), title='data')
    plt.add(ift.log10(mask.adjoint(signal_response.force(pos))), title='signal_response')
    plt.add((ift.abs(mask.adjoint(signal_response.force(pos))-data_field)), title = "Residual")
    plt.output(ny =2 , nx = 4, xsize= 30, ysize= 15, name='map.png')
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
