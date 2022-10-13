import numpy as np
import nifty8 as ift
import xubik0 as xu


npix_s = 1024
fov = 4
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])
position_space = ift.makeDomain(position_space)
coords = xu.coord_center(1024, 8)
z = np.zeros([1024, 1024])
for u in coords:
    co_x, co_y = np.unravel_index(u, [npix_s, npix_s])
    print(co_x,co_y)
    z[co_x, co_y] = 10000
zf = ift.makeField(position_space, z)


psf_file = np.load("data/npdata/psf_patches/4952_patches_v2.npy", allow_pickle=True).item()

psf_file = psf_file["psf_sim"]
psfs = []
for p in psf_file:
    p_arr = p.val
    p_arr[p_arr<50] = 0
    p = ift.makeField(p.domain, p_arr)
    norm_val = p.integrate().val**-1
    norm = ift.ScalingOperator(p.domain, norm_val)
    psf_norm = norm(p)
    psfs.append(psf_norm.val)
psfs = np.array(psfs, dtype="float64")

np.save("data/npdata/psf_patches/4952_patches_10e6_cut_50.npy", psfs)
conv_op = xu.OverlapAddConvolver(position_space, psfs, 64, 64)

res = conv_op(zf)

xu.plot_single_psf(zf, "testplot_10e6_symlog_v0.png", logscale=True, vmin=0)
xu.plot_single_psf(res, "test_plot2_10e6_symlog_v0.png", logscale=True, vmin=0)
