import numpy as np

from matplotlib.colors import LogNorm
import nifty8 as ift
import xubik0 as xu

ift.random.push_sseq_from_seed(15)

inhomogen = True
cfg = xu.get_cfg("config.yaml")

npix_s = 1024
fov = 4.0
position_space = ift.makeDomain(ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s]))

points = ift.InverseGammaOperator(position_space, **cfg['points'])
signal = points.ducktape("points").real.ducktape_left("signal")
# PSF
if inhomogen:
    psf_file = np.load("../data/npdata/psf_patches/4952_patches_v2.npy", allow_pickle=True).item()["psf_sim"]
    psfs = []
    for p in psf_file:
        psfs.append(p.val)
    psfs = np.array(psfs, dtype="float64")
    conv = xu.OverlapAddConvolver(position_space, psfs, 64, 128)
    cut = xu.MarginZeroPadder(position_space, ((conv.target.shape[0] - position_space.shape[0])//2), space=0).adjoint
    response = cut @ conv
    response = response.ducktape("signal")

else:
    psf_arr = np.load("../data/npdata/df_4952_observation.npy", allow_pickle=True).item()["psf_sim"].val[:, :, 0]
    max_idx = np.unravel_index(np.argmax(psf_arr), position_space.shape)
    psf_arr = np.roll(psf_arr, (-max_idx[0], -max_idx[1]), axis=(0,1))
    psf_field = ift.Field.from_raw(position_space, psf_arr)
    norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
    psf = norm(psf_field)
    signal_prox = ift.ScalingOperator(psf.domain, 1).ducktape("signal")
    response = xu.convolve_field_operator(psf, signal_prox)

# Data
data = np.load("../data/npdata/synth_data.npy", allow_pickle=True).item()
data_field = ift.Field.from_raw(position_space, data.val)
# likelihood = ift.PoissonianEnergy(data_field) @ signal_response
exit()
mock = np.zeros(position_space.shape)
mock[50, 50] = 190800
mock[500, 500] = 190800
mock[700, 700] = 190800
mock = ift.makeField(position_space, mock)
fa = ift.FieldAdapter(position_space,"signal").adjoint
mock = fa(mock)
mock_sr = response(mock)
factor = data_field.val[499,499]/mock_sr.val[499,499]
print("GT value: ",mock_sr.val[499,499])
print("Factor: ",factor)
print("Residual: ", data_field.val[499,499]-mock_sr.val[499,499])
mock_sr_scaled = mock_sr*factor
print("Scaled Value: ", mock_sr_scaled.val[499,499])
res = data_field-mock_sr_scaled
print("No residual: ", res.val[499,499])

xu.plot_single_psf(data_field, "synth_data_log.png", logscale=True, vmin=1)
xu.plot_single_psf(mock_sr, f"app_gt_log_ih_{inhomogen}.png", logscale=True, vmin=1)
xu.plot_single_psf(ift.abs(res), f"res_{inhomogen}.png", logscale=True, vmin=1)
exit()
# End of Loop
ic_sampling = ift.AbsDeltaEnergyController(**cfg['ic_sampling'])
ic_sampling_nl = ift.AbsDeltaEnergyController(**cfg['ic_sampling_nl'])
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)
minimizer_sampling = None

ic_newton = ift.AbsDeltaEnergyController(**cfg['ic_newton'])
minimizer = ift.NewtonCG(ic_newton)

pos = 0.1 * ift.from_random(signal.domain)
transpose = xu.Transposer(signal.target)

nl_sampling_minimizer = cfg['Nsamples']
global_it = cfg['global_it']
n_samples = cfg['Nsamples']
samples = ift.optimize_kl(
    likelihood,
    global_it,
    n_samples,
    minimizer,
    ic_sampling,
    minimizer_sampling,
    plottable_operators={
        "signal": transpose@signal,
    },
    output_directory="synth_rec_inhom",
    initial_position=pos,
    comm=xu.library.mpi.comm,
    overwrite=True,
    resume=True
)
