import numpy as np
from jubik0.instruments.jwst.rotation_and_shift.nufft_rotation_and_shift import (
    build_nufft_rotation_and_shift)
from jax import config, devices


config.update('jax_default_device', devices('cpu')[0])


def test_field_sameaxis():
    maxx, maxy = 256, 256
    field = np.zeros((maxx, maxy))
    field[100:150, 100:150] = 1
    field[200:250, 100:150] = 1

    xx, yy = np.array(np.meshgrid(np.arange(0, maxx, 1),
                                  np.arange(0, maxy, 1)))
    rs = build_nufft_rotation_and_shift(
        1, 1, field.shape, (1., 1.), field.shape)
    field_mapped = rs(field, np.array((xx, yy)))

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(field, origin='lower')
    # axes[1].imshow(field_mapped, origin='lower')
    # plt.show()

    assert np.allclose(field, field_mapped, atol=1e-4)


# def test_field_differentaxis():
#     maxx, maxy = 256, 356
#     field = np.zeros((maxx, maxy))
#     field[100:150, 100:150] = 1
#     field[200:250, 100:150] = 1
#
#     xx, yy = np.array(np.meshgrid(np.arange(0, maxx, 1),
#                                   np.arange(0, maxy, 1)))
#     rs = build_nufft_rotation_and_shift(
#         1, 1, field.shape, (1., 1.), field.shape)
#     field_mapped = rs(field, np.array((xx, yy)))
#
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(field, origin='lower')
#     axes[1].imshow(field_mapped, origin='lower')
#     plt.show()
#
#     assert np.allclose(field, field_mapped, atol=1e-4)
