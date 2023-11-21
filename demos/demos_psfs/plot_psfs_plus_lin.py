
# %%
import jubik0 as ju
import nifty8 as ift
import numpy as np

from matplotlib.colors import LogNorm
# %%

dir_path = "data/psf_info/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]

args = {'cmap': 'BuGn', 'norm': LogNorm(vmin=1E-7, vmax=0.012)}

file = dir_path + fname[0]
obs = ju.eROSITA_PSF(file)
obs.plot_psfs("psf_plots", **args)

energy = '3000'
pointing_center = (1800, 1800)
fov = (3600, 3600)

npix = (512, 512)
dists = tuple(ff/pp for ff, pp in zip(fov, npix))
domain = ift.RGSpace(npix, distances=dists)

psf_func = obs.psf_func_on_domain(energy, pointing_center, domain)

c2params = {'npatch': 8, 'margfrac': 0.062, 'want_cut': False}
op2 = obs.make_psf_op(energy, pointing_center, domain,
                      conv_method='LIN', conv_params=c2params)


# %%
def get_kernels_and_sources(domain, psf_func):
    rnds = np.zeros(domain.shape)
    rnds[18::40, 18::40] = 1.
    cc = (np.arange(ss)*dd for ss,dd in zip(domain.shape, domain.distances))
    cc = np.meshgrid(*cc, indexing='ij')
    cx, cy = cc[0], cc[1]
    inds = np.where(rnds != 0.)
    cx = cx[inds]
    cy = cy[inds]

    res = np.zeros_like(rnds)
    for ix, iy, xx, yy in zip(inds[0],inds[1], cx, cy):
        psf = psf_func(xx, yy) * domain.scalar_dvol
        psf = np.roll(psf, ix, axis = 0)
        psf = np.roll(psf, iy, axis = 1)
        res += psf
    rnds = ift.makeField(domain, rnds)
    return ift.makeField(domain, res), rnds

kernels, sources = get_kernels_and_sources(domain, psf_func)

res = op2(sources)

# %%
args = {'cmap': 'BuGn', 'norm': LogNorm(vmin=1E-5, vmax=0.2)}
ju.plot_result(kernels, outname="psf_plots/interpolated_kernel.png",
               title="Interpolated PSFs[tm1_2dpsf_190219v05.fits] from CALDB",
               **args)
ju.plot_result(res, outname="psf_plots/lin_kernel.png",
               title="Patching and linear interpolation response",
               **args)
ju.plot_result((res-kernels).abs(), outname="psf_plots/diff_kernel.png",
               title="Absolute difference response",
               **args)

# %%
