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


from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

import nifty8.re as jft
from .utils import load_fits


@dataclass
class ImageData:
    data_2d: ArrayLike
    noise_2d: ArrayLike
    mask: ArrayLike
    response: Optional[jft.Model] = None
    response_no_psf: Optional[jft.Model] = None
    pixel_size: float = 1

    def plot_data(self):
        """Plots source, data, convergence and deflection images."""
        import matplotlib.pyplot as plt

        plot_titles = ['Data', 'Mask', 'Noise']
        plot_data = [
            self.data_2d,
            self.mask,
            self.noise_2d
        ]
        fig, axes = plt.subplots(1, 3)
        for data, title, ax in zip(plot_data, plot_titles, axes.flatten()):
            im = ax.imshow(data, origin='lower')
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
        plt.show()


def load_jwst_data(
        cfg_data: dict,
) -> ImageData:
    '''Load data from file'''

    data_2d = load_fits(cfg_data['data_path'])
    noise_2d = load_jwst_std(cfg_data, data_2d)
    mask = load_jwst_mask(cfg_data, data_2d, noise_2d)

    pixel_size = cfg_data['pixel_size']
    if np.isscalar(pixel_size):
        pixel_size = (float(pixel_size),) * len(data_2d.shape)
    else:
        tmp = np.empty(len(data_2d.shape), dtype=float)
        tmp[:] = pixel_size
        pixel_size = tuple(tmp)

    # Some tests
    assert data_2d.shape == noise_2d.shape
    assert data_2d.shape == mask.shape

    return ImageData(
        data_2d=data_2d,
        noise_2d=noise_2d,
        mask=mask,
        pixel_size=pixel_size
    )


def load_jwst_mask(data_cfg: dict, data: ArrayLike, noise: ArrayLike) -> ArrayLike:
    """Loads the mask from the configuration file."""
    if data_cfg.get('mask_path', None) is None:
        return np.logical_or(
            np.isnan(data),
            np.isnan(noise))
    else:
        mask = load_fits(data_cfg['mask_path'])
        mask = mask.astype(bool)
        nan_mask = np.logical_or(  # Mask where data or noise is nan
            np.isnan(data),
            np.isnan(noise)
        )
        return np.logical_or(mask, nan_mask)


def load_jwst_std(data_cfg: dict, data: ArrayLike) -> Tuple[ArrayLike, float]:
    """Loads the noise from the configuration file."""
    if data_cfg.get('noise_path', None) is None:
        noise_scale = data_cfg['noise_scale']
        return np.full(data.shape, noise_scale)
    else:
        return load_fits(data_cfg['noise_path'])
