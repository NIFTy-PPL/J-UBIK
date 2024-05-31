import numpy as np
import jax.numpy as jnp
import nifty8 as ift

from jubik0.operators.convolve_utils import (psf_convolve_operator,
                                             psf_lin_int_operator, get_psf)

# def test_psf():
#     import pylab as plt

#     max_radec = (2.,2.)
#     npix_x, npix_y = 128, 128

#     ra = np.arange(npix_x) / npix_x * max_radec[0]
#     ra -= 0.5*max_radec[0]
#     dec = np.arange(npix_y) / npix_y * max_radec[1]
#     dec -= 0.5*max_radec[1]
#     ra, dec = ra[1:], dec[1:]
#     #print(ra)
#     #print(dec)


#     sig = 0.1
#     def func(r, dx,dy):
#         tm = 3.*dy**2 - dx**2 - dy
#         dr = np.sqrt((dx/(1.+2.*r**2))**2 + tm**2)
#         return np.exp(-0.5*(dr/sig)**2)

#     rs = np.array([0., 0.1, 0.5, 0.7, 1.])

#     nx = 128*2
#     ny = 128*2
#     dra = np.linspace(-1.,1., num = nx)
#     ddec = np.linspace(-1.,1., num = ny)
#     dx = dra[1] - dra[0]
#     dy = ddec[1] - ddec[0]
#     dra, ddec = np.meshgrid(dra, ddec, indexing='ij')
#     #dradecs = np.stack((dra, ddec), axis = -1)

#     psfs = list([func(rr, dra, ddec) for rr in rs])
#     psfs = np.stack(psfs, axis = 0)

#     #for pp,rr in zip(psfs,rs):
#     #    plt.imshow(pp.T, origin='lower')
#     #    plt.title(f'radius = {rr}')
#     #    plt.show()

#     patch_centers = np.outer(np.ones_like(rs), np.array([128, 128]))
#     patch_deltas = (dx, dy)
#     center = (1.,1.)

#     func_psf = get_psf(psfs, rs, patch_centers, patch_deltas, center)
#     #func_psf = get_psf(rs, dradecs, psfs, center, max_radec)

#     ddra, dddec = jnp.meshgrid(ra, dec, indexing='ij')
#     ra, dec = 1.5, .5
#     mypsf = func_psf(ra, dec, ddra, dddec)
#     im = plt.imshow(mypsf.T, origin='lower', vmin = 0., vmax = 1.)
#     plt.colorbar(im)
#     plt.show()

#     test_r = np.sqrt((ra-center[0])**2 + (dec-center[1])**2)
#     gtpsf = func(test_r, ddra, dddec)
#     im = plt.imshow(gtpsf.T, origin='lower', vmin = 0., vmax = 1.)
#     plt.colorbar(im)
#     plt.show()

# def compare_psf_ops():
#     sig = 0.1
#     def func(r, dx,dy):
#         #tm = 3.*dy**2 - dx**2 - dy
#         fct = (1.+3.*r**2)
#         dr = np.sqrt((dx/fct)**2 + dy**2)
#         #dr = (dx**2 + dy**2)
#         dr /= 0.01
#         return np.exp(-0.5*(dr/sig)**2) / fct**2

#     rs = np.array([0., 0.1, 0.5, 0.7, 1.])

#     nx = 256
#     ny = 256
#     dra = np.linspace(-1.,1., num = nx)
#     ddec = np.linspace(-1.,1., num = ny)
#     dx = dra[1] - dra[0]
#     dy = ddec[1] - ddec[0]
#     dra, ddec = np.meshgrid(dra, ddec, indexing='ij')
#     psfs = list([func(rr, dra, ddec) for rr in rs])
#     psfs = np.stack(psfs, axis = 0)

#     patch_centers = np.outer(np.ones_like(rs), np.array([nx//2, ny//2]))
#     patch_deltas = (dx, dy)
#     center = (1.,1.)

#     shp = (200, 200)
#     domain = ift.RGSpace(shp, tuple(2./ss for ss in shp))

#     psf_infos = {'psfs' : psfs,
#                  'rs' : rs,
#                  'patch_center_ids' : patch_centers,
#                  'patch_deltas' : patch_deltas,
#                  'pointing_center' : center}
#     msc_infos = {'base' : (3,3), 'min_baseshape' : (5,5), 'linlevel' : (1,1),
#                  'kernel_sizes' : ((5,5),(3,3)),
#                  'keep_overlap' : ((False,False),(True,True),(False,False)),
#                  'local_kernel' : (True, True)}
#     msc_op = psf_convolve_operator(domain, psf_infos, msc_infos)

#     int_op = psf_lin_int_operator(domain, 10, psf_infos, margfrac=0.1)
#     int_cut = int_op._cut
#     msc_op = int_cut @ msc_op

#     rnd = ift.from_random(int_op.domain)

#     res1 = msc_op(rnd)
#     res2 = int_op(rnd)

#     import timeit
#     for _ in range(10):
#         t0 = timeit.default_timer()
#         res1 = msc_op(rnd)
#         t1 = timeit.default_timer()
#         print("MSC: ", t1-t0)
#         t0 = timeit.default_timer()
#         res2 = int_op(rnd)
#         t1 = timeit.default_timer()
#         print("Lin: ", t1-t0)

#     pl = ift.Plot()
#     pl.add(res1, title = 'Msc')
#     pl.add(res2, title = 'int')
#     pl.output(nx=2)

# compare_psf_ops()
