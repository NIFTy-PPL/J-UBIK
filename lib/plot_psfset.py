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


fileloader = np.load("strainset_psf.npy", allow_pickle=True).item()

psf = fileloader["psf_sim"]
plot_single_psf(psf[0] + 1, "logplot_psf.png")

psfset = psf[0]
for i in range(8):
    psfset = psfset + psf[i + 1]

psfset = psfset + 1
plot_single_psf(psfset, "psfset.png")
