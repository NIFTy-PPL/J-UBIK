import jax
import numpy as np
import jubik0 as ju
import pytest

import matplotlib.pyplot as plt
# TODO Test against old implementation
shape = (36, 36)
n_patches = 16
overlap = 2

x = np.ones(shape)

a = ju.slice_patches(x, shape, n_patches, overlap)


def f(field):
    """Slice patcher partly evaluated."""
    return ju.slice_patches(field, shape, n_patches, overlap)


fadj = jax.linear_transpose(f, x)

# Plotting
plt.imshow(x)
plt.colorbar()
plt.show()

res = f(x)
back = fadj(res)[0]

plt.imshow(back)
plt.colorbar()
plt.show()

# TODO Add adjointness test
a = np.random.rand(shape[0], shape[1])
b = np.random.rand(res.shape[0], res.shape[1], res.shape[2])

# forward
res1 = np.vdot(b, f(a))
# backward
res2 = np.vdot(fadj(b), a)

np.allclose(res1, res2)
