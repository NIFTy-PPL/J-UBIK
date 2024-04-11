import jubik0 as ju
import nifty8 as ift
import numpy as np

from jax import config
from jax import random
config.update('jax_enable_x64', True)

ift.set_nthreads(8)

seed = 42
key = random.PRNGKey(seed)


def get_kernels_and_sources(domain, psf_func):
    rnds = np.zeros(domain.shape)
    rnds[::20, ::20] = 1.
    cc = (np.arange(ss)*dd for ss,dd in zip(domain.shape, domain.distances))
    cc = np.meshgrid(*cc, indexing='ij')
    cx, cy = cc[0], cc[1]
    inds = np.where(rnds != 0.)
    cx = cx[inds]
    cy = cy[inds]

    res = np.zeros_like(rnds)
    for ix, iy, xx, yy in zip(inds[0],inds[1], cx, cy):
        psf = psf_func(xx, yy) * domain.scalar_dvol
        psf = np.roll(psf, ix, axis=0)
        psf = np.roll(psf, iy, axis=1)
        res += psf
    rnds = ift.makeField(domain, rnds)
    return ift.makeField(domain, res), rnds

# dirs
dir_path = "data/psf_info/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]
filename = dir_path + fname[0]

# PSF Object
obs = ju.eROSITA_PSF(filename)

# more numbers about the observation
energy = ['3000']
# FIXME Convention for Pointing Center (Array / List ?)
pointing_center = np.array([[1800, 1800]])
fov = (1, 3600, 3600)
npix = (1, 512, 512)
dists = tuple(ff/pp for ff, pp in zip(fov, npix))

domain_nifty = ift.RGSpace(npix, distances=dists)
domain_jubik = ju.Domain(npix, dists)
rnds = ift.from_random(domain_nifty)

c2params = {'npatch': 8, 'margfrac': 0.062, 'want_cut': False}

op1 = obs.make_psf_op(energy, pointing_center, domain_jubik,
                      conv_method='LINJAX', conv_params=c2params)
res1 = op1(rnds.val)

# TODO FIX LIN for 3D
# op2 = obs.make_psf_op(energy,
#                       pointing_center,
#                       domain_nifty,
#                       conv_method='LIN',
#                       conv_params=c2params)
# res2 = op2(rnds).val

# print("")
# print("Equality of LIN and LINJAX: ", np.allclose(res1, res2))

# Test alternative to get the operator
test_psf = ju.build_erosita_psf([filename],
                                energy, pointing_center,
                                domain_jubik,
                                c2params["npatch"],
                                c2params["margfrac"],
                                c2params["want_cut"])
res3 = test_psf(rnds.val)
print("Equality of other LINJAX instance: ", np.allclose(res1, res3))

# Jax Lin with Energy dimension
print("Check JaxLin with broadcasting over energies")

# TODO add benchmarks for performace and do further tests

#FIXME Check why the energies are in a weird order when we get them from eROSITA_PSF
energies = ['0277', '0930', '1486', '3000', '4508', '6398', '8040']
fov = (1, 3600, 3600)
npix = (7, 512, 512)
dists = tuple(ff/pp for ff, pp in zip(fov, npix))

domain_nifty_2 = ift.RGSpace(npix, distances=dists)
domain_jubik_2 = ju.Domain(npix, dists)
rnds_3d = ift.from_random(domain_nifty_2)

op3 = obs.make_psf_op(energies, pointing_center, domain_jubik_2,
                      conv_method='LINJAX', conv_params=c2params)

res1 = op3(rnds_3d.val)

points = np.zeros(domain_jubik_2.shape)
points[:,::20, ::20] = 1.

res_points = op3(points)

def plotter(arr):
    import matplotlib.pyplot as plt
    for i in range(arr.shape[0]):
        plt.imshow(arr[i])
        plt.show()

print("Success")
