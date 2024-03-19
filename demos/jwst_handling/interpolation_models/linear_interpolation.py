from jax.scipy.ndimage import map_coordinates
from jax import vmap

from functools import partial


def build_linear_interpolation(
    subsample_centers, mask, order=3, updating=False
):
    interpolation = partial(
        # map_coordinates, order=order, mode='constant', cval=0.0)
        map_coordinates, order=order, mode='wrap')
    interpolation = vmap(interpolation, in_axes=(None, 0))

    if mask is not None:
        subsample_centers = subsample_centers[:, :, mask]

    if updating:
        def integration(x):
            field, xy_shift = x
            out = interpolation(
                field, subsample_centers - xy_shift[None, :, None])
            return out.sum(axis=0) / out.shape[0]

    else:
        def integration(x):
            out = interpolation(x, subsample_centers)
            return out.sum(axis=0) / out.shape[0]

    return integration
