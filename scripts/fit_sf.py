import numpy as np

import nifty8 as ift
import xubik0 as xu

ift.set_nthreads(2)

cfg = xu.get_cfg("scripts/config.yaml")

prefix = cfg["prefix"]
npix_s = 1024  # number of spacial bins per axis
fov = 4.0
energy_bin = 2
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

#Model P(s)
diffuse = ift.SimpleCorrelatedField(position_space, **cfg['priors_diffuse'])
pspec = diffuse.power_spectrum
diffuse = diffuse.exp()
points = ift.InverseGammaOperator(position_space, **cfg['points'])
points = points.ducktape("points")

signal = points + diffuse
signal = signal.real
signal_dt = signal.ducktape_left('full_signal')

#Likelihood P(d|s)
signal_fa = ift.FieldAdapter(signal_dt.target['full_signal'], 'full_signal')
likelihood_list = []

exp_norm_max, exp_norm_mean, exp_norm_std = xu.get_norm_exposure_patches(cfg['datasets'], position_space, 1)
print(f'Mean of exposure-map-norm: {exp_norm_mean} \nStandard deviation of exposure-map-norm: {exp_norm_std}')
for dataset in cfg['datasets']:
    #Loop
    observation = np.load("data/npdata/"+prefix+"_"+str(dataset)+"_observation.npy", allow_pickle=True).item()

    #PSF
    psf_file = np.load("data/npdata/psf_patches/"+str(dataset)+"_patches.npy", allow_pickle=True).item()["psf_sim"]
    psfs = []
    for p in psf_file:
        psfs.append(p.val)
    psfs = np.array(psfs, dtype="float64")

    # psf_arr = observation['psf_sim'].val[:, :, energy_bin]
    # psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
    # psf_field = ift.Field.from_raw(position_space, psf_arr)
    # norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
    # psf = norm(psf_field)

    #Data
    data = observation["data"].val[:, :, energy_bin]
    data_field = ift.Field.from_raw(position_space, data)
    #Likelihood
    conv_op = xu.OAnew.force(signal_fa.target, psfs, 64, 16)
    convolved = conv_op @ signal_fa
    cut = xu.MarginZeroPadder(convolved.target, ((position_space.shape[0] -convolved.target.shape[0])//2), space=0).adjoint

    #Exp
    exp = observation["exposure"].val[:, :, energy_bin]
    exp_field = ift.Field.from_raw(position_space, exp)
    if dataset == cfg['datasets'][0]:
        norm_first_data = xu.get_norm(exp_field, data_field)
    normed_exp_field = exp_field * norm_first_data
    # norm = xu.get_norm(exp_field, data_field)
    # norm_list.append(norm)
    # normed_exp_field = ift.Field.from_raw(position_space, exp) *norm
    normed_exp_field = cut(normed_exp_field)
    normed_exposure = ift.makeOp(normed_exp_field)
    #Mask
    mask = xu.get_mask_operator(normed_exp_field)


    signal_response = mask @ normed_exposure @ convolved
    cut_data = cut(data_field)
    masked_data = mask(cut_data)
    likelihood = ift.PoissonianEnergy(masked_data) @ signal_response
    likelihood.name = dataset
    likelihood_list.append(likelihood)

likelihood_sum = likelihood_list[0]
for i in range(1, len(likelihood_list)):
    likelihood_sum = likelihood_sum + likelihood_list[i]
likelihood_sum = likelihood_sum(signal_dt)


samples = ift.ResidualSampleList.load("perseus_rec_0/pickle/last")#, comm=comm)

# Plotting
m = 0
fits_list =["perseus_rec_0"]#, "perseus_rec_1"]

elim = (0.5, 10)
dist = np.exp(np.log(elim[1]/elim[0])/4)
ranges = []
a = elim[0]
for k in range(4):
    ranges.append((a, a*dist))
    a = a * dist
ranges = np.around(ranges, decimals=1)
print(ranges)


mean, var = samples.sample_stat(diffuse)
folder= 'perseus_0_'

fig, ax = plt.subplots(figsize=(6,6), dpi=300)
im = ax.imshow(mean.val[256:768, 256:768], origin="lower",extent=(-2,2,-2,2), vmin=0.01, vmax=10)#  interpolation='none')
ax.set_xlabel("FOV [arcmin]")
ax.set_ylabel("FOV [arcmin]")
ax.set_title("Diffuse Emission, post. mean, sat. linear scale,"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.savefig(folder+'diffuse_mean.png')


fig, ax = plt.subplots(figsize=(6,6), dpi=300)
im=ax.imshow(var.sqrt().val[256:768, 256:768], origin="lower",extent=(-2,2,-2,2),vmax=10)
ax.set_xlabel("FOV [arcmin]")
ax.set_ylabel("FOV [arcmin]")
ax.set_title("Diffuse Emission, posterior std, linear scale,"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.savefig(folder+'diffuse_std.png')

mean, var = samples.sample_stat(signal)

fig, ax = plt.subplots(figsize=(6,6), dpi=300)
im = ax.imshow(mean.val[256:768, 256:768], origin="lower",extent=(-2,2,-2,2), vmin=0.01, vmax=10)#  interpolation='none')
ax.set_xlabel("FOV [arcmin]")
ax.set_ylabel("FOV [arcmin]")
ax.set_title("Sky Emission, post. mean, sat. linear scale,"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.savefig(folder+'sky_mean.png')


fig, ax = plt.subplots(figsize=(6,6), dpi=300)
im=ax.imshow(var.sqrt().val[256:768, 256:768], origin="lower",extent=(-2,2,-2,2), vmin = 0.01, vmax=10)
ax.set_xlabel("FOV [arcmin]")
ax.set_ylabel("FOV [arcmin]")
ax.set_title("Sky Emission, posterior std, linear scale,"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.savefig(folder+'sky_std.png')

exit()

# End of Loop
ic_newton = ift.AbsDeltaEnergyController(**cfg['ic_newton'])
ic_sampling = ift.AbsDeltaEnergyController(**cfg['ic_sampling'])
ic_sampling_nl = ift.AbsDeltaEnergyController(**cfg['ic_sampling_nl'])
minimizer = ift.NewtonCG(ic_newton)

nl_sampling_minimizer = None#ift.NewtonCG(ic_sampling_nl)

pos = 0.1 * ift.from_random(signal.domain)
transpose = xu.Transposer(signal.target)

global_it = cfg['global_it']
n_samples = cfg['Nsamples']
samples = ift.optimize_kl(
    likelihood_sum,
    global_it,
    n_samples,
    minimizer,
    ic_sampling,
    nl_sampling_minimizer,
    export_operator_outputs={
        "signal": transpose@signal,
        "point_sources": transpose@points,
        "diffuse": transpose@diffuse,
        "power_spectrum": pspec,
        #" inverse_gamma_q": points.q(),
    },
<<<<<<< HEAD
    output_directory="../df_rec",
=======
    output_directory="perseus_rec_2",
>>>>>>> last work
    initial_position=pos,
    comm=xu.library.mpi.comm,
    resume=True
)

