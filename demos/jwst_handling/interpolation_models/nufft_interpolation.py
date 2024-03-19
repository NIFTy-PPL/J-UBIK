from jax_finufft import nufft2
from jax import vmap
from jax.numpy.fft import ifftshift, ifft2

from numpy import pi


def build_nufft_interpolation(subsample_centers, mask, shape):

    if mask is not None:
        subsample_centers = subsample_centers[:, :, mask]

    xy_finufft = subsample_centers / shape[None, :, None] * 2 * pi

    def interpolation(field, coords):
        return nufft2(field, coords[0], coords[1])

    interpolation = vmap(interpolation, in_axes=(None, 0))

    def integration(field):
        f_field = ifftshift(ifft2(field))
        # out = nufft2(f_field, xy_finufft[:, 0, :], xy_finufft[:, 1, :]).real
        out = interpolation(f_field, xy_finufft).real
        return out.sum(axis=0) / out.shape[0]

    return integration
