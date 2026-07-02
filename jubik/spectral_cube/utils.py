import numpy as np

from .axes import CubeAxes


def broadcast_to_full(array: np.ndarray, in_axes: int | tuple, n_axis):
    if not isinstance(in_axes, (tuple, list)):
        in_axes = (in_axes,)

    if len(array.shape) != len(in_axes):
        raise ValueError(f"Number of in_axes must match {array.ndim}")

    if len(set(in_axes)) != len(in_axes):
        raise ValueError("in_axes must be unique")

    if any(k < 0 or k >= n_axis for k in in_axes):
        raise ValueError("Axis out of bounds")

    shp = [1] * n_axis

    for axis, size in zip(in_axes, array.shape):
        shp[axis] = size

    return array.reshape(shp)


def remove_singleton_axes(array: np.ndarray, out_axes: int | tuple):
    if not isinstance(out_axes, (tuple, list)):
        out_axes = (out_axes,)

    if len(set(out_axes)) != len(out_axes):
        raise ValueError("in_axes must be unique")

    if any(k < 0 or k >= array.ndim for k in out_axes):
        raise ValueError("Axis out of bounds")

    out_axes = set(out_axes)
    slicer = []

    for axis, size in enumerate(array.shape):
        if axis in out_axes:
            slicer.append(slice(None))
        elif size == 1:
            slicer.append(0)
        else:
            raise ValueError(
                f"Cannot remove a nonsingleton axis: axis {axis}, size {size}"
            )

    return array[tuple(slicer)]


def integrate_cube(cube, axes, deltas):
    de = broadcast_to_full(deltas, axes, n_axis=6)
    return np.sum(cube * de, axis=axes, keepdims=True)


def intensity_weighted_spectral_moments(order, cube, spectral_centers, spectral_widths):
    # Computes the spectral mean (order = 1) or higher central statistical moments (order > 1)
    m0 = integrate_cube(cube, CubeAxes.SPECTRAL, spectral_widths)
    m1 = (
        integrate_cube(cube, CubeAxes.SPECTRAL, spectral_centers * spectral_widths) / m0
    )

    if order < 1:
        raise ValueError("Order has to be a positive integer")

    if order == 1:
        return m1
    else:
        del_v = (
            broadcast_to_full(spectral_centers, CubeAxes.SPECTRAL, n_axis=len(m1.shape))
            - m1
        )
        return (
            integrate_cube(cube * del_v**order, CubeAxes.SPECTRAL, spectral_widths) / m0
        )


def intensity_weighted_standardized_spectral_moments(
    order, cube, spectral_centers, spectral_widths
):
    if order < 3:
        raise ValueError(
            "Can only compute standardized moments of at least third order."
        )

    cm = intensity_weighted_spectral_moments(
        order=order,
        cube=cube,
        spectral_centers=spectral_centers,
        spectral_widths=spectral_widths,
    )

    m2 = intensity_weighted_spectral_moments(
        order=2,
        cube=cube,
        spectral_centers=spectral_centers,
        spectral_widths=spectral_widths,
    )

    return cm / m2 ** (order / 2)
