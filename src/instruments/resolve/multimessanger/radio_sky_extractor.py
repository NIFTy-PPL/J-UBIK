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
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Julian Rüstig

from ....grid import Grid
from ....color import ColorRanges
from ..constants import RESOLVE_SKY_UNIT

import nifty8.re as jft

import jax
import jax.numpy as jnp

import astropy.units as u

from typing import Union


def build_extract_sky(sky_key: str | None):
    """Extract the sky by key."""
    if sky_key is None:
        return lambda x: x
    return lambda x: x[sky_key]


def build_radio_slicing(index_of_last_radio_bin: int | None):
    """Build a Grid corresponding to the radio part of the sky.

    ***Warning: The radio part of  the sky is assumed to be in the first
    indices of the sky.***

    Parameters
    ----------
    index_of_last_radio_bin: int | None
        The index of the last radio bin. We assume that the radio sky is in the
        first part of sky frequencies.
    """

    # TODO: Make the radio sclicing independent from the assumption that the
    # first indices correspond to the radio part of the sky.

    if index_of_last_radio_bin is None:
        return lambda x: x

    def radio_sky_extractor(tree):
        return jax.tree.map(lambda x: x[: index_of_last_radio_bin + 1], tree)

    return radio_sky_extractor


def build_unit_conversion(sky_unit: u.Unit | None):
    """Convert sky to `sky_unit`."""
    if sky_unit is None:
        return lambda x: x

    conversion_factor = sky_unit.to(RESOLVE_SKY_UNIT)

    def unit_conversion(tree):
        return jax.tree.map(lambda x: x * conversion_factor, tree)

    return unit_conversion


def resolve_transpose(tree):
    """Transpose the spatial axis."""

    return jax.tree.map(lambda x: jnp.transpose(x, (0, 1, 2, 4, 3)), tree)


# TODO : This function shouldn't be here but part of the sky models.
def build_radiofy_sky(sky_domain_shape: tuple[int]):
    """Make the output shape of the sky conform to the standard axis:
    (polarization, time, frequencies, space, space)

    Parameters
    ----------
    sky_domain_shape: tuple[int]
        The shape of the sky domain, i.e. the SkyModel.target.shape.
    """

    if len(sky_domain_shape) == 2:
        return lambda tree: jax.tree.map(lambda x: x[None, None, None], tree)

    elif len(sky_domain_shape) == 3:
        return lambda tree: jax.tree.map(lambda x: x[None, None], tree)

    elif len(sky_domain_shape) == 4:
        return lambda tree: jax.tree.map(lambda x: x[None], tree)

    elif len(sky_domain_shape) == 5:
        return lambda tree: jax.tree.map(lambda x: x, tree)

    else:
        raise ValueError(f"Shape: {sky_domain_shape} is not compatible with radio sky.")


class RadioSkyExtractor(jft.Model):
    def __init__(
        self,
        domain: Union[jft.ShapeWithDtype, dict],
        extract_sky: callable,
        slice_radio_sky: callable,
        unit_conversion: callable,
        radiofy: callable,
        transpose: callable,
    ):
        self.extract = extract_sky
        self.slicing = slice_radio_sky
        self.conv = unit_conversion
        self.radiofy = radiofy
        self.trans = transpose
        super().__init__(domain=domain)

    def __call__(self, x):
        return self.trans(self.radiofy(self.conv(self.slicing(self.extract(x)))))


def build_radio_sky_extractor(
    index_of_last_radio_bin: int | None,
    sky_model: jft.Model,
    sky_key: str | None = "sky",
    sky_unit: u.Unit | None = None,
) -> RadioSkyExtractor:
    """Builds a jft.Model that extracts the bins from the sky_model, which
    correspond to the radio sky and converts the sky from its unit system to
    the resolve unit system [Jy/rad].

    If no index_of_last_radio_bin is provided the sky_model is just passed on,
    but it's expected that the `sky_model` target has no keys.

    ***Warning: The radio part of  the sky is assumed to be in the first
    indices of the sky.***

    Parameters
    ----------
    index_of_last_radio_bin: int | None
        The index of the last radio bin. We assume that the radio sky is in the
        first part of sky.
    sky_model: jft.Model
        The model of the sky, it's target domain is the input domain of the
        radio sky extractor.
    sky_key: str | None
        The potential key of the sky.
    sky_unit: u.Unit | None
        The unit of the sky.
    """

    extract, slicing, conv, trans = (
        build_extract_sky(sky_key),
        build_radio_slicing(index_of_last_radio_bin),
        build_unit_conversion(sky_unit),
        resolve_transpose,
    )
    radiofy = build_radiofy_sky(extract(sky_model.target).shape)

    return RadioSkyExtractor(sky_model.target, extract, slicing, conv, radiofy, trans)


def build_radio_grid(
    index_of_last_radio_bin: int | None,
    sky_grid: Grid,
):
    """Build a Grid corresponding to the radio part of the sky.

    ***Warning: The radio part of  the sky is assumed to be in the first
    indices of the sky.***

    Parameters
    ----------
    index_of_last_radio_bin: int | None
        The index of the last radio bin. We assume that the radio sky is in the
        first part of sky.
    sky_grid: Grid
        The Grid holding the sky information.
    """

    # TODO: Make the radio sclicing independent from the assumption that the
    # first indices correspond to the radio part of the sky.

    if index_of_last_radio_bin is None:
        return sky_grid

    return Grid(
        spatial=sky_grid.spatial,
        spectral=ColorRanges(sky_grid.spectral[: index_of_last_radio_bin + 1]),
    )
