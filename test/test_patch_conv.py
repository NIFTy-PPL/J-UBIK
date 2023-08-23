import nifty8 as ift
import numpy as np
import xubik0 as xu

import matplotlib.pyplot as plt

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

# nifty implementation
old_conv = xu.OAnew.cut_force(domain, psfs, n_patches, margin, False)
res1 = old_conv(test_f)

# plot nifty result
plt.imshow(res1.val)
plt.colorbar()
plt.show()

plt.clf()

# jax result
psfs_prep = np.load("debugging_kernel.npy", allow_pickle=True).item().val
res2 = xu.linpatch_convolve(test_f.val, shape, psfs_prep,
                            n_patches_per_axis, margin)
cres = res2 * domain.scalar_dvol

plt.imshow(cres)
plt.colorbar()
plt.show()
print("Results of two methods coincide:", np.allclose(cres, res1.val))
