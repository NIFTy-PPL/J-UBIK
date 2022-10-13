import numpy as np
import nifty8 as ift
import astropy.io.fits as ast
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import ndimage, misc

# Plots
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

m = 0
for folder in fits_list:
        diffuse_mean = ast.open(folder+'/diffuse/last_mean.fits')[0].data
        points_mean = ast.open(folder+'/point_sources/last_mean.fits')[0].data
        std = ast.open(folder+'/diffuse/last_std.fits')[0].data
        points_mean[points_mean<1.4]=0
        mean = points_mean + diffuse_mean
        mean = mean[256:768, 256:768]
        std = std[256:768, 256:768]
        diffuse_mean = diffuse_mean[256:768, 256:768]
        fig, ax = plt.subplots(sharey=True, nrows=2, figsize=(8,12), dpi=300)
        ax[0].imshow(diffuse_mean, origin="lower",extent=(-2,2,-2,2), vmin=10**-1, vmax=7)
        ax[0].set_title("Reconstructed diffuse emission (sat. linear scale)")
        ax[0].set_xlabel("FOV [arcmin]")
        ax[0].set_ylabel("FOV [arcmin]")
        ax[1].imshow(mean, origin="lower",extent=(-2,2,-2,2),  norm=LogNorm())#,  interpolation='none')
        ax[1].set_title("Reconstructed Image (logscale)")
        ax[1].set_xlabel("FOV [arcmin]")
        fig.suptitle('Perseus Cluster between '+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ', fontsize=16)
        plt.savefig(folder+'.png')

        fig, ax = plt.subplots(figsize=(6,6), dpi=300)
        ax.imshow(mean, origin="lower",extent=(-2,2,-2,2), vmin=0.01, vmax=10)#  interpolation='none')
        ax.set_xlabel("FOV [arcmin]")
        ax.set_ylabel("FOV [arcmin]")
        ax.set_title("Reconstructed Image saturated (linear scale)"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
        plt.savefig(folder+'_b.png')

        fig, ax = plt.subplots(figsize=(6,6), dpi=300)
        ax.imshow(diffuse_mean, origin="lower",extent=(-2,2,-2,2), vmin=0.01, vmax=10)#  interpolation='none')
        ax.set_xlabel("FOV [arcmin]")
        ax.set_ylabel("FOV [arcmin]")
        ax.set_title("Reconstructed Diffuse Emission, saturated linear scale"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
        plt.savefig(folder+'_diffuse.png')

        # fig, ax = plt.subplots(figsize=(6,6), dpi=300)
        # result = ndimage.gaussian_gradient_magnitude(diffuse_mean, sigma=0.5)
        # ax.imshow(result, origin="lower",extent=(-2,2,-2,2),vmax=0.5)#  interpolation='none')
        # ax.set_xlabel("FOV [arcmin]")
        # ax.set_ylabel("FOV [arcmin]")
        # ax.set_title("Reconstructed Image saturated (linear scale)"+ str(ranges[m][0]) + '-' + str(ranges[m][1]) + ' keV ')
        # plt.savefig(folder+'_diffuse_ggm.png')

        m+=1
