import jax
import numpy as np
import xubik0 as xu
import matplotlib.pyplot as plt

# TODO Calculate Valid Shapes
shape = (64, 64)
n_patches = 4
overlap = 0

x = np.ones(shape)
kernel = np.ones(shape)
oa, padded = xu.linpatch_convolve(x, shape, kernel, n_patches, overlap)
res = oa(padded)[0]

print(res.shape)
plt.imshow(res)
plt.colorbar()
plt.show()
