import numpy as np
import matplotlib.pyplot as plt

v = np.ones([20, 20])
tgtshape = [100,100]
for d in [0, 1]:
    shp = list(v.shape)
    idx = (slice(None),) * d
    shp[d] = tgtshape[d]

    xnew = np.zeros(shp)
    Nyquist = v.shape[d]//2
    i1 = idx + (slice(0, Nyquist+1),)
    xnew[i1] = v[i1]
    i1 = idx + (slice(None, -(Nyquist+1), -1),)
    xnew[i1] = v[i1]
    v = xnew

x = np.zeros([100,100])
x[54, 54] = 1
print(np.max(x))
x[v == 0] = 0
print(np.max(x))
plt.imshow(x)
plt.show()
