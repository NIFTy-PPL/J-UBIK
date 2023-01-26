import os

import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from .utils import get_data_domain
from ..library.chandra_observation import ChandraObservationInformation
import astropy as ao


def plot_slices(field, outname, logscale=False):
    img = field.val
    npix_e = field.domain.shape[-1]
    nax = np.ceil(np.sqrt(npix_e)).astype(int)
    half_fov = field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0 / 60. # conv to arcmin
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-half_fov, half_fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm()

    fig, ax = plt.subplots(
        nax, nax, figsize=(11.7, 8.3), sharex=True, sharey=True, dpi=200
    )
    ax = ax.flatten()
    for ii in range(npix_e):
        im = ax[ii].imshow(img[:, :, ii], **pltargs)
        cb = fig.colorbar(im, ax=ax[ii])
    fig.tight_layout()
    if outname != None:
        fig.savefig(outname)
    plt.close()


def plot_result(field, outname, logscale=False, **args):
    fig, ax = plt.subplots(dpi=300, figsize=(11.7, 8.3))
    img = field.val
    half_fov = field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0 / 60 # conv to arcmin
    pltargs = {"origin": "lower", "cmap": "viridis", "extent": [-half_fov, half_fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm()
    pltargs.update(**args)
    im = ax.imshow(img, **pltargs)
    cb = fig.colorbar(im)
    fig.tight_layout()
    if outname != None:
        fig.savefig(outname)
    plt.close()


def plot_fused_data(obs_info, img_cfg, obslist, outroot, center=None):
    grid = img_cfg["grid"]
    data_domain = get_data_domain(grid)
    data = []
    for obsnr in obslist:
        info = ChandraObservationInformation(obs_info["obs" + obsnr], **grid, center=center)
        data.append(info.get_data(f"./data_{obsnr}.fits"))
    full_data = sum(data)
    full_data_field = ift.makeField(data_domain, full_data)
    plot_slices(full_data_field, outroot + "_full_data.png")


def plot_rgb_image(file_name_in, file_name_out, log_scale=False):
    import astropy.io.fits as pyfits
    from astropy.visualization import make_lupton_rgb
    import matplotlib.pyplot as plt
    color_dict = {0: "red", 1: "green", 2: "blue"}
    file_dict = {}
    for key in color_dict:
        file_dict[color_dict[key]] = pyfits.open(f"{file_name_in}_{color_dict[key]}.fits")[0].data
    rgb_default = make_lupton_rgb(file_dict["red"], file_dict["green"], file_dict["blue"], filename=file_name_out)
    if log_scale:
        plt.imshow(rgb_default, norm=LogNorm(), origin='lower')
    else:
        plt.imshow(rgb_default, origin='lower')


def plot_image_from_fits(file_name_in, file_name_out, log_scale=False):
    import matplotlib.pyplot as plt
    from astropy.utils.data import get_pkg_data_filename
    from astropy.io import fits
    image_file = get_pkg_data_filename(file_name_in)
    image_data = fits.getdata(image_file, ext=0)
    plt.figure()
    plt.imshow(image_data, norm=LogNorm())
    plt.savefig(file_name_out)


def plot_single_psf(psf, outname, logscale=True, vmin=None, vmax=None):
    half_fov = psf.domain[0].distances[0] * psf.domain[0].shape[0] / 2.0 / 60 # conv to arcmin
    psf = psf.val  # .reshape([1024, 1024])
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-half_fov, half_fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots()
    psf_plot = ax.imshow(psf, **pltargs)
    fig.colorbar(psf_plot)
    fig.tight_layout()
    fig.savefig(outname, dpi=1500)
    plt.close()


def plot_psfset(fname, outname, npix, n, in_one=True):
    fileloader = np.load(fname, allow_pickle=True).item()
    psf = fileloader["psf_sim"]
    if in_one:
        psfset = psf[0]
        for i in range(1, n ** 2):
            psfset = psfset + psf[i]
        plot_single_psf(psfset, outname + "psfset.png", logscale=True)

    else:
        p = ift.Plot()
        for k in range(10):
            p.add(psf[k], title=f"{k}", norm=LogNorm())
        p.output(name=outname + "psfs.png", xsize=20, ysize=20)


def _append_key(s, key):
    if key == "":
        return s
    return f"{s} ({key})"


def _plot_samples(filename, samples, plotting_kwargs):
    samples = list(samples)

    if isinstance(samples[0].domain, ift.DomainTuple):
        samples = [ift.MultiField.from_dict({"": ss}) for ss in samples]
        # if ground_truth is not None:
        #     ground_truth = ift.MultiField.from_dict({"": ground_truth})
    if not all(isinstance(ss, ift.MultiField) for ss in samples):
        raise TypeError
    keys = samples[0].keys()

    p = ift.Plot()
    for kk in keys:
        single_samples = [ss[kk] for ss in samples]

        if ift.plot.plottable2D(samples[0][kk]):
            # if ground_truth is not None:
            # p.add(ground_truth[kk], title=_append_key("Ground truth", kk),
            #       **plotting_kwargs)
            # p.add(None)
            for ii, ss in enumerate(single_samples):
                # if (ground_truth is None and ii == 16) or (ground_truth is not None and ii == 14):
                #     break
                p.add(ss, title=_append_key(f"Sample {ii}", kk), **plotting_kwargs)
        else:
            n = len(samples)
            alpha = n * [0.5]
            color = n * ["maroon"]
            label = None
            # if ground_truth is not None:
            #     single_samples = [ground_truth[kk]] + single_samples
            #     alpha = [1.] + alpha
            #     color = ["green"] + color
            #     label = ["Ground truth", "Samples"] + (n-1)*[None]
            p.add(single_samples, color=color, alpha=alpha, label=label,
                  title=_append_key("Samples", kk), **plotting_kwargs)
    p.output(name=filename)


def _plot_stats(filename, op, sl, plotting_kwargs):
    try:
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    mean, var = sl.sample_stat(op)
    p = ift.Plot()
    # if op is not None: TODO: add Ground Truth plotting capabilities
    #     p.add(op, title="Ground truth", **plotting_kwargs)
    p.add(mean, title="Mean", **plotting_kwargs)
    p.add(var.sqrt(), title="Standard deviation", **plotting_kwargs)
    p.output(name=filename, ny=2)
    # print("Output saved as {}.".format(filename))


def plot_sample_and_stats(output_directory, operators_dict, sample_list, iterator, plotting_kwargs):
    for key in operators_dict:
        op = operators_dict[key]
        results_path = os.path.join(output_directory, key)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        filename = os.path.join(output_directory, key, "stats_{}.png".format(iterator))
        filename_samples = os.path.join(output_directory, key, "samples_{}.png".format(iterator))

        _plot_stats(filename, op, sample_list, plotting_kwargs)
        _plot_samples(filename_samples, sample_list.iterator(op), plotting_kwargs)


def create_output_directory(directory_name):
    output_directory = os.path.join(os.path.curdir, directory_name)
    return output_directory
