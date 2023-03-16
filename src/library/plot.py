import os
import math

import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from .utils import get_data_domain, get_cfg, create_output_directory
from ..library.sky_models import SkyModel
from ..library.erosita_response import load_erosita_response
from ..library.chandra_observation import ChandraObservationInformation


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
    rgb_default = make_lupton_rgb(file_dict["red"], file_dict["green"], file_dict["blue"],
                                  filename=file_name_out)
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


def plot_energy_slices(field, file_name, title=None, plot_kwargs={}):
    """
    Plots the slices of a 3-dimensional field along the energy dimension.

    Parameters:
    ----------
    field : ift.Field
        The field to plot.
    file_name : str
        The name of the file to save the plot.
    title : str or None
        The title of the plot. Default is None.
    plot_kwargs : `dict` keyword arguments for plotting.
        If True, the plot uses a logarithmic scale. Default is False.

    Raises:
    -------
    ValueError : if the domain of the field is not as expected.

    Returns:
    --------
    None
    """
    domain = field.domain
    if not isinstance(domain, ift.DomainTuple) or len(domain[0].shape) !=2:
        raise ValueError(f"Expected DomainTuple with the first space"
                         f"being a 2-dim RGSpace, but got {domain}")

    if len(domain) == 2 and len(domain[1].shape) != 1:
        raise ValueError(f"Expected DomainTuple with the second space"
                         f"being a 1-dim RGSpace, but got {domain}")

    if len(domain) == 1:
        p = ift.Plot()
        p.add(field, **plot_kwargs)
        p.output(name=file_name)

    elif len(domain) == 2:
        p = ift.Plot()
        for i in range(field.shape[2]):
            slice = ift.Field(ift.DomainTuple.make(domain[0]), field.val[:, :, i])
            p.add(slice, title=f'{title}_e_bin={i}', **plot_kwargs)
        p.output(name=file_name)
    else:
        raise NotImplementedError


def plot_energy_slice_overview(field_list, field_name_list, file_name, title=None, logscale=False):
    """
    Plots a list of fields in one plot separated by energy bins

    Parameters:
    ----------
    field_list : List of ift.Fields
                 The field to plot.
    file_name : str
        The name of the file to save the plot.
    title : str or None
        The title of the plot. Default is None.
    logscale : bool
        If True, the plot uses a logarithmic scale. Default is False.

    Raises:
    -------
    ValueError : if the domain of the field is not as expected.
    ValueError: If the number of field names does not match the number of fields.

    Returns:
    --------
    None
    """
    domain = field_list[0].domain
    if any(field.domain != domain for field in field_list):
        raise ValueError('All fields need to have the same domain.')

    if not isinstance(domain, ift.DomainTuple) or len(domain[0].shape) != 2:
        raise ValueError(f"Expected DomainTuple with the first space "
                         f"being a 2-dim RGSpace, but got {domain}")

    if len(domain) == 2 and len(domain[1].shape) != 1:
        raise ValueError(f"Expected DomainTuple with the second space "
                         f"being a 1-dim RGSpace, but got {domain}")

    if len(field_list) != len(field_name_list):
        raise ValueError("Every field needs a name")

    pltargs = {"origin": "lower", "cmap": "cividis"}
    if logscale:
        pltargs["norm"] = LogNorm()
    cols = math.ceil(math.sqrt(len(field_list)))  # Calculate number of columns
    rows = math.ceil(len(field_list) / cols)
    if len(domain) == 1:
        if len(field_list) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11.7, 8.3),
                                   sharex=True, sharey=True, dpi=200)
            im = ax.imshow(field_list[0].val, **pltargs)
            ax.set_title(f'{title}_{field_name_list[0]}')
        else:
            fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(11.7, 8.3),
                                   sharex=True, sharey=True, dpi=200)
            ax = ax.flatten()
            for i, field in enumerate(field_list):
                im = ax[i].imshow(field.val, **pltargs)
                ax[i].set_title(f'{title}_{field_name_list[i]}')
        fig.tight_layout()
        fig.savefig(f'{file_name}.png')
        plt.close()
    elif len(domain) == 2:
        for i in range(domain[1].shape[0]):
            if len(field_list) == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11.7, 8.3),
                                       sharex=True, sharey=True, dpi=200)
                im = ax.imshow(field_list[0].val, **pltargs)
                ax.set_title(f'{title}_{field_name_list[0]}')
            else:
                fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(11.7, 8.3),
                                       sharex=True, sharey=True, dpi=200)
                ax = ax.flatten()
                for j, field in enumerate(field_list):
                    im = ax[j].imshow(field.val[:, :, i], **pltargs)
                    ax[j].set_title(f'{field_name_list[j]}')
            fig.tight_layout()
            fig.savefig(f'{file_name}_e_bin={i}.png')
            plt.close()
    else:
        raise NotImplementedError


def plot_erosita_priors(seed, n_samples, config_path, response_path, priors_dir,
                        plotting_kwargs=None, common_colorbar=False):
    priors_dir = create_output_directory(priors_dir)
    cfg = get_cfg(config_path)  # load config

    if plotting_kwargs is None:
        plotting_kwargs = {}

    if 'norm' in plotting_kwargs:
        norm = plotting_kwargs.pop('norm')
        norm = type(norm)
    else:
        norm = None

    sky_dict = SkyModel(config_path).create_sky_model()
    plottable_ops = sky_dict.copy()

    # Loads random seed for mock positions
    ift.random.push_sseq_from_seed(seed)
    positions = []
    for sample in range(n_samples):
        positions.append(ift.from_random(plottable_ops['sky'].domain))

    plottable_samples = plottable_ops.copy()
    for key, val in plottable_samples.items(): # FIXME: refactor into a function
        plottable_samples[key] = [val.force(pos) for pos in positions]

    for key, val in plottable_samples.items():
        if common_colorbar: # FIXME: not working
            vmin = min(np.min(val[i].val) for i in range(n_samples))
            vmax = max(np.max(val[i].val) for i in range(n_samples))
        else:
            vmin = vmax = None
        p = ift.Plot()
        for i in range(n_samples):
            p.add(val[i], vmin=vmin, vmax=vmax, norm=norm(),
                  title=key + ' prior', **plotting_kwargs)
            if 'title' in plotting_kwargs:
                del (plotting_kwargs['title'])
        filename = priors_dir + f'priors_{key}.png'
        p.output(name=filename,
                 **plotting_kwargs)
        print(f'Prior signal saved as {filename}.')

    if response_path is not None:  # FIXME: when R will be pickled, load from file
        tm_ids = cfg['telescope']['tm_ids']
        plottable_ops.pop('pspec')

        resp_dict = load_erosita_response(config_path, priors_dir)

        for tm_id in tm_ids:
            tm_key = f'tm_{tm_id}'
            R = resp_dict[tm_key]['mask'].adjoint @ resp_dict[tm_key]['R']
            plottable_samples = {}

            for key, val in plottable_ops.items():
                SR = R @ val
                plottable_samples[key] = [SR.force(pos) for pos in positions]

            for key, val in plottable_samples.items():
                if common_colorbar:
                    vmin = min(np.min(val[i].val) for i in range(n_samples))
                    vmax = max(np.max(val[i].val) for i in range(n_samples))
                else:
                    vmin = vmax = None
                p = ift.Plot()
                for i in range(n_samples):
                    p.add(val[i], vmin=vmin, vmax=vmax, norm=norm(),
                          title=tm_key + ' ' + key + ' prior signal response', **plotting_kwargs)
                    if 'title' in plotting_kwargs:
                        del (plotting_kwargs['title'])
                res_path = priors_dir + f'tm{tm_id}/'
                filename = res_path + f'sr{tm_id}_priors_{key}.png'
                p.output(name=filename, **plotting_kwargs)
                print(f'Signal response saved as {filename}.')
