import numpy as np
from scipy.stats import wasserstein_distance

# from nifty8.re.caramel import power_analyze

import jax.numpy as jnp
from nifty8.re.correlated_field import get_fourier_mode_distributor
import numpy as np


def _single_power_analyze(field, distances):
    # super slow but works

    mode_id, mode_dist, mode_mul = get_fourier_mode_distributor(
        field.shape, distances)

    tmp = np.zeros(mode_mul.shape)
    for ii in range(len(mode_mul)):
        tmp[ii] = field[mode_id == ii].mean()
    return tmp


def power_analyze(field, distances, keep_phase_information=False, return_distances=False):

    field_real = not jnp.any(jnp.iscomplex(field))

    if keep_phase_information:
        parts = [field.real*field.real, field.imag*field.imag]
    else:
        if field_real:
            parts = [field**2]
        else:
            parts = [field.real*field.real + field.imag*field.imag]

    parts = [_single_power_analyze(part, distances)
             for part in parts]

    return parts[0] + 1j*parts[1] if keep_phase_information else parts[0]


def source_distortion_ratio(input_source, model_source):
    return 10 * np.log10(np.linalg.norm(input_source) /
                         np.linalg.norm(input_source - model_source))


def cross_correlation(input, recon):
    return np.fft.ifft2(
        np.fft.fft2(input).conj() * np.fft.fft2(recon)
    ).real.max()


def get_power(field, mask, reshape):
    return power_analyze(
        np.fft.fft2(field[mask].reshape((reshape,)*2)),
        (1.0,)*2  # FAKE DISTANCES ::FIXME::
    )


def chi2(data, model, std):
    ''' Computes the chi2 of the model compared to the data.'''
    return np.nansum(((data - model)/std)**2)


def redchi2(data, model, std, dof):
    ''' Computes the reduced chi2 of the model compared to the data.'''
    return chi2(data, model, std) / dof


def wmse(data, model, std):
    ''' Computes the MSE (mean square error) in units of variance.'''
    return np.nanmean(((data - model)/std)**2)


def find_corners(xy_positions):
    xy_positions = np.array(xy_positions)

    maxx, maxy = np.argmax(xy_positions, axis=1)
    minx, miny = np.argmin(xy_positions, axis=1)

    square_corners = np.array([
        xy_positions[:, maxx],
        xy_positions[:, maxy],
        xy_positions[:, minx],
        xy_positions[:, miny],
    ])
    return square_corners


def point_in_polygon(point, polygon):
    """Determine if the point (x, y) is inside the given polygon.
    polygon is a list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def pixels_in_rectangle(corners, shape):
    """Get a boolean numpy array where each element indicates whether the
    center of the pixel at that index is inside the rectangle."""

    # Convert corners to a list of tuples
    polygon = [(corners[i][0], corners[i][1]) for i in range(corners.shape[0])]

    pixel_map = np.zeros(shape, dtype=bool)

    min_x = np.floor(np.min(corners[:, 0]))
    max_x = np.ceil(np.max(corners[:, 0]))
    min_y = np.floor(np.min(corners[:, 1]))
    max_y = np.ceil(np.max(corners[:, 1]))

    start_x = max(0, int(min_x))
    end_x = min(shape[1], int(max_x) + 1)
    start_y = max(0, int(min_y))
    end_y = min(shape[0], int(max_y) + 1)

    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            if point_in_polygon((x + 0.5, y + 0.5), polygon):
                pixel_map[x, y] = True

    return pixel_map


def build_evaluation_mask(reco_grid, data_set, comp_sky=None):
    evaluation_mask = np.full(reco_grid.shape, False)

    for data_key in data_set.keys():
        _, data_grid = data_set[data_key]['data'], data_set[data_key]['grid']

        index = data_grid.wcs.index_grid_from_wl_extrema(
            data_grid.world_extrema())
        wl_data_centers = data_grid.wcs.wl_from_index([index])[0]

        px_reco_datapix_cntr = reco_grid.wcs.index_from_wl(wl_data_centers)[0]
        corners = find_corners(px_reco_datapix_cntr.reshape(2, -1))
        tmp_mask = pixels_in_rectangle(corners, reco_grid.shape)
        evaluation_mask += tmp_mask

        if comp_sky is not None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2)
            ax, ay = axes
            ax.imshow(comp_sky)
            ax.scatter(*corners.T, color='orange')
            ay.imshow(tmp_mask)
            ay.scatter(*corners.T, color='orange')
            plt.show()

    return evaluation_mask


def smallest_enclosing_mask(pixel_map):
    """Finds the edge points of the largest and smallest `True` pixels in specified sections of the pixel_map."""
    assert pixel_map.shape[0] == pixel_map.shape[1]

    mask = np.zeros_like(pixel_map)

    shape = pixel_map.shape[0]
    for ii in range(shape):
        if np.all(pixel_map[ii:shape-ii, ii:shape-ii]):
            mask[ii:shape-ii, ii:shape-ii] = True
            return ii, mask

    return ii, mask
