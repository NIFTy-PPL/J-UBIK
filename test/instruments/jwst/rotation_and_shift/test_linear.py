import numpy as np
from jubik0.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import (
    build_linear_rotation_and_shift,
)


def test_field_sameaxis_ij():
    maxx, maxy = 256, 256
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xy = np.array(
        np.meshgrid(np.arange(0, maxx, 1), np.arange(0, maxy, 1), indexing="ij")
    )
    rs = build_linear_rotation_and_shift(indexing="ij", order=1)
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-5)


def test_field_sameaxis_xy():
    maxx, maxy = 256, 256
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xy = np.array(
        np.meshgrid(np.arange(0, maxx, 1), np.arange(0, maxy, 1), indexing="xy")
    )
    rs = build_linear_rotation_and_shift(indexing="xy", order=1)
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-5)


def test_field_differentaxis_ij():
    maxx, maxy = 256, 325
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xy = np.array(
        np.meshgrid(np.arange(0, maxx, 1), np.arange(0, maxy, 1), indexing="ij")
    )
    rs = build_linear_rotation_and_shift(indexing="ij", order=1)
    field_mapped = rs(field, xy)
    assert np.allclose(field, field_mapped, atol=1e-4)


def test_field_differentaxis_xy():
    maxx, maxy = 256, 325
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xy = np.array(
        np.meshgrid(np.arange(0, maxx, 1), np.arange(0, maxy, 1), indexing="xy")
    )
    rs = build_linear_rotation_and_shift(indexing="xy", order=1)
    field_mapped = rs(field, xy)

    assert np.allclose(field, field_mapped, atol=1e-4)
