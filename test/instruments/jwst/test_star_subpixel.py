import numpy as np
import pytest

from jubik.instruments.jwst.alignment.star_alignment import Star


class _IdentityWcs:
    """World coordinates are native detector pixel coordinates (x, y)."""

    def world_to_pixel(self, position):
        return np.asarray(position[0], float), np.asarray(position[1], float)


@pytest.mark.parametrize("s", [1, 2, 3])
def test_native_pixel_center_maps_to_fine_pixel_center(s):
    x, y = Star.skycoord_to_subpixel((0.0, 0.0), _IdentityWcs(), subsample_factor=s)
    assert (x, y) == ((s - 1) / 2, (s - 1) / 2)


@pytest.mark.parametrize("s, expected", [(1, 4.0), (2, 8.5), (3, 13.0)])
def test_subpixel_with_cutout_origin(s, expected):
    # Detector (x, y) = (7, 5) in a cutout starting at row 1, column 3 -> (4, 4).
    x, y = Star.skycoord_to_subpixel(
        (7.0, 5.0), _IdentityWcs(), min_row=1, min_column=3, subsample_factor=s
    )
    assert (x, y) == (expected, expected)
