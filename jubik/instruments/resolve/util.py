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
# Copyright(C) 2020-2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import nifty.cl as ift
import numpy as np
import jax
import jax.numpy as jnp

from astropy import units as u
from astropy.coordinates import SkyCoord


def cast_to_dtype(tree, dtype=jnp.float32):
    return jax.tree.map(lambda x: x.astype(dtype), tree)


def my_assert(*conds):
    if not all(conds):
        raise RuntimeError


def my_asserteq(*args):
    for aa in args[1:]:
        if args[0] != aa:
            raise RuntimeError(f"{args[0]} != {aa}")


def my_assert_isinstance(*args):
    args = list(args)
    cls = args.pop()
    for aa in args:
        if not isinstance(aa, cls):
            raise RuntimeError(aa, cls)


def compare_attributes(obj0, obj1, attribute_list):
    return all(_fancy_equal(getattr(obj0, a), getattr(obj1, a)) for a in attribute_list)


def _fancy_equal(o1, o2):
    if not _types_equal(o1, o2):
        return False

    # Turn MultiField into dict
    if isinstance(o1, ift.MultiField):
        o1, o2 = o1.val, o2.val

    # Compare dicts
    if isinstance(o1, dict):
        return _deep_equal(o1, o2)

    # Compare simple objects and np.ndarrays
    return _compare_simple_or_array(o1, o2)


def _deep_equal(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError

    if a.keys() != b.keys():
        return False

    return all(_compare_simple_or_array(a[kk], b[kk]) for kk in a.keys())


def _compare_simple_or_array(a, b):
    equal = a == b
    if isinstance(equal, np.ndarray):
        return np.all(equal)
    assert isinstance(equal, bool)
    return equal


def _types_equal(a, b):
    return type(a) == type(b)


def is_single_precision(dtype):
    if isinstance(dtype, dict):
        return any(is_single_precision(vv) for vv in dtype.values())
    if dtype in [np.float32, np.complex64]:
        return True
    elif dtype in [np.float64, np.complex128]:
        return False
    raise TypeError(f"DType {dtype} is not a floating point dtype.")


def dtype_float2complex(dt):
    if dt == np.float64:
        return np.complex128
    if dt == np.float32:
        return np.complex64
    raise ValueError


def dtype_complex2float(dt, force=False):
    if dt == np.complex128:
        return np.float64
    if dt == np.complex64:
        return np.float32
    if force:
        if dt in [np.float32, np.float64]:
            return dt
    raise ValueError


def calculate_phase_offset_to_image_center(
    sky_center: SkyCoord,
    phase_center: SkyCoord,
) -> tuple[float | None, float | None]:
    """Calculate the relative shift of the phase center to the sky center
    (reconstruction center) in radians.

    Parameters
    ----------
    sky_center: astropy.SkyCoord
        The world coordinate of the sky center.
    phase_center: astropy.SkyCoord
        The world coordinate of the phase center of the observation.
    """
    r = sky_center.separation(phase_center)
    phi = sky_center.position_angle(phase_center)
    # FIXME: center (x, y) switch maybe because of the ducc0 fft?
    center_y = r.to(u.rad).value * np.cos(phi.to(u.rad).value)
    center_x = r.to(u.rad).value * np.sin(phi.to(u.rad).value)

    if np.isnan(center_x) or np.isnan(center_y):
        return 0.0, 0.0

    return center_x, center_y
