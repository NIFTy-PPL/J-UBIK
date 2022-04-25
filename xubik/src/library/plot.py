import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm


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


def plot_single_psf(psf, outname, logscale=True, vmin=None, vmax=None):
    fov = psf.domain[0].distances[0] * psf.domain[0].shape[0] / 2.0
    psf = psf.val #.reshape([1024, 1024])
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-fov, fov] * 2}
    if logscale == True:
        pltargs["norm"] = SymLogNorm(1, vmin=vmin, vmax=vmax)
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
        for i in range(1, n**2):
            psfset = psfset + psf[i]
        plot_single_psf(psfset, outname + "psfset.png", logscale=True)

    else:
        p = ift.Plot()
        for k in range(10):
            p.add(psf[k], title=f"{k}", norm=LogNorm())
        p.output(name=outname +"psfs.png", xsize=20, ysize=20)
