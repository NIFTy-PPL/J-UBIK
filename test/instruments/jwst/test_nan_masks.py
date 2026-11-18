"""The extra nan-mask files: their names carry the cutout bounds they were made for, and a mask for other bounds is
refused."""

import numpy as np
import pytest

from jubik.instruments.jwst.parse.masking.data_mask import NanMasks

# SPT2147's 12-exposure F2100W mask, the longest name in use (202 characters of bounds)
SPT2147 = (
    "357_475_689_808__509_627_684_803__489_607_540_659__372_490_562_681__392_510_704_823__544_662_699_818__"
    "524_642_555_674__407_525_577_696__427_545_719_838__579_697_714_833__559_677_570_689__442_560_592_711"
)


def _bounds(name: str) -> list[tuple[int, ...]]:
    return [tuple(int(b) for b in bound.split("_")) for bound in name.split("__")]


def _load(tmp_path, bounds, prefix="f115w_b"):
    """Save a mask the way sptLensing's jwst_masks driver names it; return the mask and the loader for ``bounds``."""
    mask = np.random.default_rng(0).random((len(bounds), 4, 5)) > 0.5
    path = tmp_path / f"{prefix}{NanMasks.get_bounds_name(bounds)}.npy"
    np.save(path, mask)
    return mask, NanMasks.from_yaml_dict({"nan_mask": {"f115w": str(path)}})


def test_names_that_fit_stay_as_written():
    assert NanMasks.get_bounds_name(_bounds(SPT2147)) == SPT2147
    assert NanMasks.get_bounds_name(
        np.array([[1036, 1290, 776, 1031], [1041, 1295, 777, 1032]])
    ) == ("1036_1290_776_1031__1041_1295_777_1032")


def test_21_exposures_save_and_load_under_a_digest(tmp_path):
    """B1938's 21 dither points: 21 four-digit bounds would name a 430-character file."""
    bounds = [(1036 + i, 1290 + i, 776 + i, 1031 + i) for i in range(21)]
    name = NanMasks.get_bounds_name(bounds)
    assert name.isdigit() and name == NanMasks.get_bounds_name(np.array(bounds))
    mask, masks = _load(tmp_path, bounds, prefix="extra_mask_f2100w_b")
    assert len(f"extra_mask_f2100w_b{name}.npy") < 255
    np.testing.assert_array_equal(masks.get("f115w", bounds).load_mask(), mask)


@pytest.mark.parametrize("n", [2, 12, 21])
def test_a_mask_for_other_bounds_is_refused(tmp_path, n):
    bounds = [(1036 + i, 1290 + i, 776 + i, 1031 + i) for i in range(n)]
    _, masks = _load(tmp_path, bounds)
    moved = [*bounds[:-1], (bounds[-1][0] + 1, *bounds[-1][1:])]
    with pytest.raises(ValueError, match="do not match"):
        masks.get("f115w", moved)
    assert (
        masks.get("f277w", moved) is None
    )  # a filter without a mask has none to check
