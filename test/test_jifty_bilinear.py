import nifty8 as ift
import nifty8.re as jft
import jubik0 as ju

import jax.numpy as jnp
import matplotlib.pyplot as plt

sp = ift.RGSpace([20, 20])
a = ju.operators.convolution_operators._bilinear_weights(sp)
b = ju.operators.jifty_convolution_operators._bilinear_weights(sp.shape[0])

plt.imshow(a.val)
plt.title("Nifty")
plt.show()

plt.clf()
plt.imshow(b)
plt.title("Jifty")
plt.show()

print(jnp.array_equal(a.val, b))
