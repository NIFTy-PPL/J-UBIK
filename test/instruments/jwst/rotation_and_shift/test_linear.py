import numpy as np
from jubik0.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import build_linear_rotation_and_shift


def test_field_sameaxis():
    maxx, maxy = 256, 256
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xx, yy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                                  np.arange(0, maxy, 1),
                                  indexing='ij')
                      )
    rs = build_linear_rotation_and_shift(1, 1)
    field_mapped = rs(field, np.array((xx, yy)))

    assert np.allclose(field, field_mapped, atol=1e-5)


def test_field_differentaxis():
    maxx, maxy = 256, 325
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xx, yy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                                  np.arange(0, maxy, 1),
                                  indexing='ij'))
    rs = build_linear_rotation_and_shift(1, 1)
    field_mapped = rs(field, np.array((xx, yy)))
    assert np.allclose(field, field_mapped, atol=1e-4)

    xx, yy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                                  np.arange(0, maxy, 1),
                                  indexing='xy'))
    rs = build_linear_rotation_and_shift(1, 1)
    field_mapped = rs(field.T, np.array((yy, xx))).T
    assert np.allclose(field, field_mapped, atol=1e-4)
