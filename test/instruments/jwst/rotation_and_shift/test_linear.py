import numpy as np
from jubik.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import (
    build_linear_rotation_and_shift,
)


def test_field_sameaxis_yx():
    maxy, maxx = 256, 256
    field = np.zeros((maxy, maxx))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    yx = np.array(
        np.meshgrid(np.arange(0, maxy, 1), np.arange(0, maxx, 1), indexing="ij")
    )
    rs = build_linear_rotation_and_shift(out_shape=field.shape, order=1)
    field_mapped = rs(field, yx)

    assert np.allclose(field, field_mapped, atol=1e-5)


def test_field_differentaxis_yx():
    maxy, maxx = 256, 325
    field = np.zeros((maxy, maxx))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    yx = np.array(
        np.meshgrid(np.arange(0, maxy, 1), np.arange(0, maxx, 1), indexing="ij")
    )
    rs = build_linear_rotation_and_shift(out_shape=field.shape, order=1)
    field_mapped = rs(field, yx)
    assert np.allclose(field, field_mapped, atol=1e-4)
