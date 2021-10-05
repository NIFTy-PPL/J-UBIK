import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_slices(field, outname, logscale=False):

    img     = field.val
    npix_e  = field.domain.shape[-1]
    nax     = np.ceil(np.sqrt(npix_e)).astype(int)
    fov     = field.domain[0].distances[0]*field.domain[0].shape[0]/2.
    pltargs = {'origin':'lower', 'cmap':'cividis', 'extent':[-fov,fov]*2}
    if logscale==True:
        pltargs['norm'] = LogNorm()

    fig, ax = plt.subplots(nax, nax, figsize=(11.7, 8.3), sharex=True, sharey=True, dpi=200)
    ax = ax.flatten()
    for ii in range(npix_e):
        im = ax[ii].imshow(img[:,:,ii], **pltargs)
        cb = fig.colorbar(im, ax=ax[ii])
    fig.tight_layout()
    if outname != None:
        fig.savefig(outname)
    #plt.show()
    plt.close()

    
def plot_result(field, outname):
    fig, ax = plt.subplots(dpi=400, figsize=(11.7, 8.3) )
    img = field.val
    fov     = field.domain[0].distances[0]*field.domain[0].shape[0]/2.# is this true?
    pltargs = {'origin':'lower', 'cmap':'hot', 'extent':[-fov,fov]*2,'norm': LogNorm()}
    im = ax.imshow(img, **pltargs)
    cb = fig.colorbar(im)
    fig.tight_layout()
    if outname != None:
        fig.savefig(outname)
    plt.close()
