import nifty8 as ift
import numpy as np
import jubik0 as ju

import pytest
import matplotlib.pyplot as plt

#FIXME: is this needed? Or can this be rewritten without actual data?

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
    thrs = 3

    # nifty result
    old_conv = ju.OAnew.cut_by_value(domain, psfs, n_patches, margin, thrs, False)
    res1 = old_conv(test_f)
    plt.imshow(res1.val)
    plt.show()

    # reload psf array
    psf_array = np.load("../../perseus/processed_data/11713.npy",
                        allow_pickle=True).item()["psf_sim"][0]
    psfs = np.array([p.val for p in psf_array])
    # cut by value
    psfs[psfs < thrs] = 0
    # jax result
    res2 = ju.linpatch_convolve(test_f.val, domain, psfs,
                                n_patches_per_axis, margin)

    plt.imshow(res2)
    plt.show()
    np.testing.assert_allclose(res2, res1.val)


if __name__ == "__main__":
    test_lin_patch_conv()
