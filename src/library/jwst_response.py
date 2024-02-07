# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Julian Ruestig


from .jwst_psf import instantiate_psf
from .utils import load_fits
from .spaces import Domain

from charm_lensing.models.parametric_models import (
    read_parametric_model, build_parametric_prior)

import nifty8.re as jft
import numpy as np
import jax.numpy as jnp

from numpy.typing import ArrayLike
from typing import Tuple, Union, Callable

from jax.scipy.signal import fftconvolve

LANCZOS_WINDOW = 3
ZERO_FLUX_KEY = 'zero_flux'


def lanczos_kernel(
    kernel_size: Tuple[int, int],
    shift: Tuple[int, int],
    lanczos_window: int = 3
) -> ArrayLike:
    """
    Function to generate a Lanczos kernel.

    Parameters:
    ----------
    kernel_size:
        The size of the square kernel.
    shift:
        A tuple with the x and y shift.
    lanczos_window:
        The size of the Lanczos window (parameter "a" in the Lanczos function).

    Returns:
    -------
    kernel: The resulting 2D Lanczos kernel as a numpy array.
    """

    # Calculate the coordinates of the center of the kernel
    center_coord = (kernel_size[0] // 2, kernel_size[1] // 2)

    # Calculate the coordinates of each pixel relative to the center
    xx, yy = np.ogrid[:kernel_size[1], :kernel_size[0]]
    relative_y_coords = yy - center_coord[1] - shift[1]
    relative_x_coords = xx - center_coord[0] - shift[0]

    # Use broadcasting to calculate the distance from each pixel to the center
    x = np.sqrt(relative_x_coords ** 2 + relative_y_coords ** 2)

    # Create the Lanczos kernel
    # Usually, lanczos_window is half of the kernel size
    a = lanczos_window
    kernel = np.sinc(x) * np.sinc(x / a)
    kernel[x > a] = 0  # Apply window function

    return kernel


def load_psf_and_lanczos_shift(
    response_cfg: dict,
    kernel_space: Domain
) -> Tuple[Union[ArrayLike, None], Tuple[int, int]]:
    '''Load data and PSF for a given key from configuration file.'''

    psf_path = response_cfg.get('psf_path', None)

    if psf_path is None:
        print('No Psf loaded')
        psf = None
        oversampling = None
    else:
        psf = load_fits(psf_path)
        psf = psf / psf.sum()

        # check if psf is oversampled, FIXME: This should be done better
        if 'oversample' in psf_path.split('/')[-1].split('.')[0]:
            oversampling = int(
                psf_path.split('/')[-1].split('.')[0].split('_')[-1]
            )
        else:
            oversampling = 1

    if response_cfg.get('pixel_shift', None) is None:
        coordinate_shift = np.array([0, 0])
    else:
        shift = np.load(response_cfg['pixel_shift'])
        pix_size = response_cfg['pixel_size']
        coordinate_shift = shift * pix_size

    if not np.logical_or(*np.isclose(coordinate_shift, 0)):
        lan_kern = lanczos_kernel(
            kernel_size=kernel_space.shape,
            shift=np.array((coordinate_shift[1], coordinate_shift[0])
                           ) / kernel_space.distances[0],
            lanczos_window=LANCZOS_WINDOW)
        lan_kern = lan_kern / np.sum(lan_kern)

        if psf is not None:
            psf_kernel = fftconvolve(lan_kern, psf, mode='same')
        else:
            psf_kernel = lan_kern
        psf = psf_kernel

    return psf, oversampling


def build_zero_flux(prefix: str, likelihood_config: dict) -> jft.Model:
    model_cfg = likelihood_config.get(ZERO_FLUX_KEY, None)
    if model_cfg is None:
        return None

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])
    zf = read_parametric_model(ZERO_FLUX_KEY)
    zfp = build_parametric_prior(zf, prefix, model_cfg)
    return jft.Model(
        lambda x: zf(zfp(x)),
        domain=zfp.domain
    )


def build_mask_operator(
    domain: jft.ShapeWithDtype,
    mask: ArrayLike
) -> jft.Model:
    assert domain.shape == mask.shape
    mask = jnp.array(mask, dtype=bool)
    return jft.Model(
        lambda x: x[mask],
        domain=domain
    )


def build_integration_operator(
        domain: Domain,
        target_distances: Tuple[float, float],
        oversampling: None | int = None
) -> Callable[[ArrayLike], ArrayLike]:
    assert len(domain.shape) == 2
    assert len(target_distances) == len(domain.distances)
    assert all([np.isclose(d, target_distances[0]) for d in target_distances])
    assert all([np.isclose(d, domain.distances[0]) for d in domain.distances])

    if target_distances[0] == domain.distances[0]:
        return lambda x: x

    # check that integration_value is integer
    downsample = [
        target_distances[ii] / domain.distances[ii]
        for ii in range(len(target_distances))
    ]
    assert all([np.isclose(d, np.round(d)) for d in downsample])

    if oversampling is not None:
        etxt = 'integration_value = {iv} != oversampling = {os}'
        assert np.isclose(downsample[0], oversampling), etxt.format(
            iv=downsample[0], os=oversampling)

    downsample = int(downsample[0])

    assert type(downsample) == int
    new_shape = (domain.shape[0] // downsample, downsample,
                 domain.shape[1] // downsample, downsample)
    reshaping_indices = np.reshape(
        np.arange(np.prod(domain.shape)), new_shape)

    def integration(x):
        y = jnp.take(x, reshaping_indices)
        return jnp.sum(y, axis=(1, 3))

    return integration


def build_jwst_response(
    domain_key: str,
    domain: Domain,
    data_pixel_size: Tuple[float, float],
    likelihood_key: str,
    likelihood_config: dict | None = None
) -> Tuple[jft.Model, jft.Model]:
    '''Build response operator

    Parameters
    ----------
    domain_key: str
        - name of the domain, needed for interfacing with the sky model.
          (Necessary to enable a single evaluation of the sky model.)
    domain: Domain - domain
    data_pixel_size: Tuple[float, float] - pixel size of the data
    likelihood_key: str - name of the likelihood
    likelihood_config: dict
        - psf_path:
            path to the psf, or None if no psf is used.
        - pixel_size (optional):
            pixel size of the data, used for the pixel shift.
        - pixel_shift (optional):
            path to the pixel shift, or None if no pixel shift is used.
            The pixel shift must be provided in units of `pixel_size`.
            It provides the shift of the data with respect to the center of the
            `domain`.

        - zero_flux (optional):
            name of the zero-flux model, or None if no zero-flux is used.
    '''

    # TODO: The check if psf, data and domain fit together can be done here.
    # PSF operator & pixel/coordinate shift
    psf_field, oversampling = load_psf_and_lanczos_shift(
        likelihood_config, domain)
    integrator = build_integration_operator(
        domain,
        data_pixel_size,
        oversampling=oversampling)
    psf = instantiate_psf(psf_field)

    # Standard response
    ptree = {domain_key: jft.ShapeWithDtype(domain.shape, float)}
    def R(x): return integrator(psf(x[domain_key]))
    def R_no_psf(x): return integrator(x[domain_key])

    # Zero-flux operator
    zf = build_zero_flux(likelihood_key, likelihood_config)
    if zf is not None:
        ptree.update(zf.domain)
        def R(x): return integrator(psf(x[domain_key])) + zf(x)
        def R_no_psf(x): return integrator(x[domain_key]) + zf(x)

    return (jft.Model(R, domain=jft.Vector(ptree)),
            jft.Model(R_no_psf, domain=jft.Vector(ptree)))
