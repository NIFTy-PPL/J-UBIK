import nifty8 as ift
import numpy as np
import xubik0 as xu

import pytest
import matplotlib.pyplot as plt

def test_lin_patch_conv():
    # load Chandra PSFS from file
    psf_array = np.load("../../perseus/processed_data/11713.npy",
                        allow_pickle=True).item()["psf_sim"][0]
    psfs = np.array([p.val for p in psf_array])

    # define domain
    shape = [512, 512]
    domain = ift.RGSpace(shape)
    test_f = ift.from_random(domain)

    # common parameters
    n_patches = 64
    n_patches_per_axis = int(np.sqrt(n_patches))
    margin = 10

    # nifty result
    old_conv = xu.OAnew.cut_force(domain, psfs, n_patches, margin, False)
    res1 = old_conv(test_f)

    plt.imshow(res1.val)
    plt.show()

    # jax result
    psfs_prep = np.load("debugging_kernel.npy", allow_pickle=True).item().val
    res2 = xu.linpatch_convolve(test_f.val, domain, psfs,
                                n_patches_per_axis, margin)

    plt.imshow(res2)
    plt.show()
    np.testing.assert_allclose(res2, res1.val)
