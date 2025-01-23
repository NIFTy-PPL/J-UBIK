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

import cProfile
import io
import pstats
from pstats import SortKey
from time import time

import nifty8 as ift
import numpy as np


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
    return all(_fancy_equal(getattr(obj0, a), getattr(obj1, a))
               for a in attribute_list)


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


def divide_where_possible(a, b):
    # Otherwise one
    if isinstance(a, ift.Field):
        dom = a.domain
        a = a.val
        dtype = a.dtype
        if isinstance(b, ift.Field):
            my_asserteq(b.dtype, a.dtype)
    elif isinstance(b, ift.Field):
        dom = b.domain
        b = b.val
        dtype = b.dtype
    else:
        raise TypeError
    arr = np.divide(a, b, out=np.ones(dom.shape, dtype=dtype), where=b != 0)
    return ift.makeField(dom, arr)


def imshow(arr, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    return ax.imshow(arr.T, origin="lower", **kwargs)


def rows_to_baselines(antenna_positions, data_field):
    ua = antenna_positions.unique_antennas()
    my_assert(np.all(antenna_positions.ant1 < antenna_positions.ant2))

    res = {}
    for iant1 in range(len(ua)):
        for iant2 in range(iant1+1, len(ua)):
            res[f"{iant1}-{iant2}"] = antenna_positions.extract_baseline(iant1, iant2, data_field)
    return res


def assert_sky_domain(dom):
    """Check that input fulfils resolve's conventions of a sky domain.

    A sky domain is a DomainTuple:

    dom = (pdom, tdom, fdom, sdom)

    where `pdom` is a `PolarizationSpace`, `tdom` and `fdom` are  an `IRGSpace`,
and `sdom` is a two-dimensional `RGSpace`.

    Parameters
    ----------
    dom : DomainTuple
        Object that is checked to fulfil the properties.
    """
    from .irg_space import IRGSpace
    from .polarization_space import PolarizationSpace

    my_assert_isinstance(dom, ift.DomainTuple)
    my_asserteq(len(dom), 4)
    pdom, tdom, fdom, sdom = dom
    for ii in range(3):
        my_asserteq(len(dom[ii].shape), 1)
    my_asserteq(len(sdom.shape), 2)
    my_assert_isinstance(pdom, PolarizationSpace)
    my_assert_isinstance(tdom, IRGSpace)
    my_assert_isinstance(fdom, IRGSpace)
    my_assert_isinstance(sdom, ift.RGSpace)
    my_asserteq(len(sdom.shape), 2)


def assert_data_domain(dom):
    """Check that input fulfils resolve's conventions of a data domain.

    A data domain is a DomainTuple:

    dom = (pdom, rdom, fdom)

    where `pdom` is a `PolarizationSpace`, `rdom` is an UnstructuredDomain representing the rows of the measurement set and
    `fdom` is an `IRGSpace` containing the frequencies.

    Parameters
    ----------
    dom : DomainTuple
        Object that is checked to fulfil the properties.
    """
    from .irg_space import IRGSpace
    from .polarization_space import PolarizationSpace

    my_assert_isinstance(dom, ift.DomainTuple)
    my_asserteq(len(dom), 3)
    pdom, rdom, fdom = dom
    for ii in range(3):
        my_asserteq(len(dom[ii].shape), 1)
    my_assert_isinstance(pdom, PolarizationSpace)
    my_assert_isinstance(rdom, ift.UnstructuredDomain)
    my_assert_isinstance(fdom, IRGSpace)


def _obj2list(obj, cls):
    if isinstance(obj, cls) or obj is None:
        return [obj]
    return list(obj)


def _duplicate(lst, n):
    if len(lst) == n:
        return lst
    if len(lst) == 1:
        return n*lst
    raise RuntimeError


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


def operator_equality(op0, op1, ntries=20, domain_dtype=np.float64, rtol=None, atol=0):
    dom = op0.domain
    if op0.domain != op1.domain:
        raise TypeError(f"Domains are incompatible.\n{op0.domain}\n\n{op1.domain}")
    if op0.target != op1.target:
        raise TypeError(f"Target domains are incompatible.\n{op0.target}\n\n{op1.target}")
    if rtol is None:
        rtol = 1e-5 if is_single_precision(domain_dtype) else 1e-11
    for ii in range(ntries):
        loc = ift.from_random(dom, dtype=domain_dtype)
        res0 = op0(loc)
        res1 = op1(loc)
        ift.extra.assert_allclose(res0, res1, rtol=rtol, atol=atol)

        linloc = ift.Linearization.make_var(loc, want_metric=True)
        res0 = op0(linloc)
        res1 = op1(linloc)
        ift.extra.assert_allclose(res0.jac(0.23*loc), res1.jac(0.23*loc), rtol=rtol, atol=atol)
        if res0.metric is not None:
            ift.extra.assert_allclose(res0.metric(loc), res1.metric(loc), rtol=rtol, atol=atol)

        tgtloc = res0.jac(0.23*loc)
        res0 = op0(linloc).jac.adjoint(tgtloc)
        res1 = op1(linloc).jac.adjoint(tgtloc)
        ift.extra.assert_allclose(res0, res1, rtol=rtol, atol=atol)
    ift.extra.check_operator(op0, loc, ntries=ntries, tol=rtol)
    ift.extra.check_operator(op1, loc, ntries=ntries, tol=rtol)


def replace_array_with_dict(arr, dct):
    """Replace all values in an array with the help of a dictionary

    Parameters
    ----------
    arr : np.ndarray
        Array that 
    dct : dict
        Dictionary that is used for translation.
    """
    k = np.array(list(dct.keys()))
    v = np.array(list(dct.values()))
    sort = k.argsort()
    k = k[sort]
    v = v[sort]
    return v[np.searchsorted(k, arr)]


def dict_inverse(dct):
    vals = dct.values()
    assert len(vals) == len(set(vals))
    return {vv: kk for kk, vv in dct.items()}
