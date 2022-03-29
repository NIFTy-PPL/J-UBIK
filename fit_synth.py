import numpy as np

import nifty8 as ift
import xubik0 as xu

inhomogen = True
cfg = xu.get_cfg("config.yaml")

npix_s = 1024
fov = 4.0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

points = ift.InverseGammaOperator(position_space, **cfg['points'])
signal = points.ducktape("points").real

# PSF
if inhomogen:
    psf_file = np.load("psf_patches/4952_patches_v2.npy", allow_pickle=True).item()["psf_sim"]
    psfs = []
    for p in psf_file:
        psfs.append(p.val)
    psfs = np.array(psfs, dtype="float64")
    conv = xu.OverlapAddConvolver(signal.target, psfs, 64, 128)
    cut = xu.MarginZeroPadder(signal.target, ((conv.target.shape[0] - signal.target.shape[0])//2), space=0).adjoint
    response = cut @ conv
    signal_response = response(signal)

else:
    psf_arr = np.load("df_4952_observation.npy", allow_pickle=True).item()["psf_sim"].val[:, :, 0]
    psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
    psf_field = ift.Field.from_raw(position_space, psf_arr)
    norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
    psf = norm(psf_field)
    signal_response = xu.convolve_field_operator(psf, signal)

# Data
data = np.load("npdata/synth_data.npy", allow_pickle=True).item()
data_field = ift.Field.from_raw(position_space, data.val)
likelihood = ift.PoissonianEnergy(data_field) @ signal_response

if inhomogen:
    mock = np.zeros(signal.target.shape)
    mock[50, 50] = 80000
    mock[530, 530] = 80000
    mock[700, 700] = 80000
    mock = ift.makeField(signal.target, mock)
    mock_sr = response(mock)

xu.plot_single_psf(data_field, "synth_data_log.png", logscale=True, vmin=1)
xu.plot_single_psf(mock_sr, "app_gt_log_old_v1.png", logscale=True, vmin=1)

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
