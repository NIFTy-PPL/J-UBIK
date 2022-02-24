import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_single_psf(psf, outname, logscale=True):
    fov = psf.domain[0].distances[0] * psf.domain[0].shape[0] / 2.0
    psf = psf.val.reshape([1024, 1024])
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-fov, fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm()
    fig, ax = plt.subplots()
    psf_plot = ax.imshow(psf, **pltargs)
    psf_cbar = fig.colorbar(psf_plot)
    fig.tight_layout()
    fig.savefig(outname, dpi=600)
    plt.close()


def coord_center(side_length, side_n):
    tdx = tdy = side_length // side_n
    xc = np.arange(tdx // 2, tdx * side_n, tdx)
    yc = np.arange(tdy // 2, tdy * side_n, tdy)
    co = np.array(np.meshgrid(xc, yc)).reshape(2, -1)
    res = np.ravel_multi_index(co, [side_length, side_length])
    return res


coords = coord_center(1024, 8)  # FIXME VARIABLES
fileloader = np.load("patches_psf.npy", allow_pickle=True).item()

psf = fileloader["psf_sim"]
source = fileloader["source"]

if False:
    p = ift.Plot()
    for k in range(16):
        p.add(psf[k], title=f"{k}", norm=LogNorm())
    p.output(name="test.png", xsize=20, ysize=20, dpi=300)

else:
    psfset = psf[0]
    sourceset =source[0]
    for i in range(63):
        psfset = psfset + psf[i + 1]
        sourceset= sourceset + source[i+1]
    plot_single_psf(psfset, "psfset.png", logscale=True)
    plot_single_psf(sourceset, "sourceset.png", logscale=True)
