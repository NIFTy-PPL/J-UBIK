import numpy as np
from jubik0.instruments.jwst.rotation_and_shift.nufft_rotation_and_shift import (
    build_nufft_rotation_and_shift)
from jax import config, devices


config.update('jax_default_device', devices('cpu')[0])


def _field_setup(maxx, maxy):
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1
    return field


def test_field_sameaxis():
    maxx, maxy = 256, 256
    field = _field_setup(maxx, maxy)

    xy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                              np.arange(0, maxy, 1),
                              indexing='ij'))
    rs = build_nufft_rotation_and_shift(
        1, 1, field.shape, (1., 1.), field.shape)
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-4)


def test_field_sameaxis_xy():
    maxx, maxy = 256, 256
    field = _field_setup(maxx, maxy)

    xy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                              np.arange(0, maxy, 1),
                              indexing='xy'))
    rs = build_nufft_rotation_and_shift(
        1, 1, field.shape, (1., 1.), field.shape, indexing='xy')
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-4)


def test_field_differentaxis():
    maxx, maxy = 256, 356
    field = _field_setup(maxx, maxy)

    xy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                              np.arange(0, maxy, 1),
                              indexing='ij'))
    rs = build_nufft_rotation_and_shift(
        1, 1, field.shape, (1., 1.), field.shape)
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-4)


def test_field_differentaxis_xy():
    maxx, maxy = 256, 356
    field = _field_setup(maxx, maxy)

    xy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                              np.arange(0, maxy, 1),
                              indexing='xy'))
    rs = build_nufft_rotation_and_shift(
        1, 1, field.shape, (1., 1.), field.shape, indexing='xy')
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-4)
