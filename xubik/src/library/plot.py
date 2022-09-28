import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .utils import get_data_domain
from ..operators.observation_operator import ChandraObservationInformation
import astropy as ao

def plot_slices(field, outname, logscale=False):
    img = field.val
    npix_e = field.domain.shape[-1]
    nax = np.ceil(np.sqrt(npix_e)).astype(int)
    fov = field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-fov, fov] * 2}
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
    # plt.show()
    plt.close()


def plot_result(field, outname, logscale=False, **args):
    fig, ax = plt.subplots(dpi=300, figsize=(11.7, 8.3))
    img = field.val
    fov = field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0  # is this true?
    pltargs = {"origin": "lower", "cmap": "viridis", "extent": [-fov, fov] * 2}
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
        info = ChandraObservationInformation(obs_info["obs"+obsnr], **grid, center=center)
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
    rgb_default = make_lupton_rgb(file_dict["red"], file_dict["green"], file_dict["blue"], minimum=0, filename = file_name_out)
    if log_scale:
        plt.imshow(rgb_default, origin='lower', norm=LogNorm())
        plt.savefig(file_name_out)
    else:
        plt.imshow(rgb_default, origin='lower')
        plt.savefig(file_name_out)

def plot_image_from_fits(file_name_in, file_name_out, log_scale=False):
    import matplotlib.pyplot as plt
    from astropy.utils.data import get_pkg_data_filename
    from astropy.io import fits
    image_file = get_pkg_data_filename(file_name_in)
    image_data = fits.getdata(image_file, ext=0)
    plt.figure()
    plt.imshow(image_data, norm= LogNorm())
    plt.savefig(file_name_out)

