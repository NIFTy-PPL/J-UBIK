from jubik.parse.wcs.spatial_model import (
    SpatialModel, yaml_dict_to_shape, yaml_dict_to_fov,
)
from jubik.parse.wcs.coordinate_system import CoordinateSystems
from jubik.wcs.frame import SpatialGeometry

from astropy.coordinates import SkyCoord
import astropy.units as u
import pytest
import numpy as np


def test_spatial_model_from_yaml_dict():
    grid_config = dict(
        shape=(384, 192),
        fov="6arcsec",
        position_angle="12.0deg",
        coordinate_frame="icrs",
        sky_center=dict(
            ra="64.66543063107049deg",
            dec="-47.86462563973049deg",
        ),
    )

    spatial_model = SpatialModel.from_yaml_dict(grid_config)

    scc = grid_config["sky_center"]
    assert spatial_model.wcs_model.center == SkyCoord(ra=scc["ra"], dec=scc["dec"])
    assert spatial_model.shape_xy == grid_config["shape"]
    assert spatial_model.fov_xy == (u.Quantity(grid_config["fov"]),) * 2
    assert spatial_model.wcs_model.position_angle == u.Quantity(
        grid_config["position_angle"]
    )
    assert spatial_model.wcs_model.coordinate_system == CoordinateSystems.icrs.value


def test_legacy_rotation_key_is_rejected():
    with pytest.raises(ValueError, match="use astronomical `position_angle`") as error:
        SpatialModel.from_yaml_dict(
            {"shape": 8, "fov": "1arcsec", "rotation": "1deg"}
        )
    assert "MR !238 (commit e995bc54)" in str(error.value)
    assert "spatial-conventions.rst" in str(error.value)


def test_square_sdim_warns_and_maps_to_shape():
    with pytest.warns(FutureWarning, match="2026-12-17"):
        parsed = yaml_dict_to_shape({"sdim": 8})
    assert parsed == (8, 8)


def test_rectangular_sdim_is_rejected():
    with pytest.raises(ValueError, match="axis order is undefined"):
        yaml_dict_to_shape({"sdim": (8, 4)})


def test_sdim_together_with_shape_is_rejected():
    with pytest.raises(ValueError, match="drop `sdim`"):
        yaml_dict_to_shape({"sdim": 8, "shape": 8})


@pytest.mark.parametrize("shape", [4, np.int64(4), (4, np.int64(6))])
def test_config_and_geometry_normalize_shape_identically(shape):
    parsed = yaml_dict_to_shape({"shape": shape})
    assert parsed == SpatialGeometry.from_xy(shape, 1 * u.arcsec).shape_xy


@pytest.mark.parametrize("shape", [True, (True, 4), (4.0, 6), (0, 4), (4, 6, 8)])
def test_config_and_geometry_reject_shape_identically(shape):
    with pytest.raises(ValueError):
        yaml_dict_to_shape({"shape": shape})
    with pytest.raises(ValueError):
        SpatialGeometry.from_xy(shape, 1 * u.arcsec)


@pytest.mark.parametrize("fov", ["1arcsec", ["1arcsec", "2arcsec"], [1, 2] * u.arcmin])
def test_config_and_geometry_normalize_fov_identically(fov):
    parsed = yaml_dict_to_fov({"fov": fov})
    assert u.allclose(parsed, SpatialGeometry.from_xy((4, 6), fov).fov_xy)


@pytest.mark.parametrize("fov", ["0arcsec", ["1arcsec", "-2arcsec"], [1, 2, 3] * u.arcsec])
def test_config_and_geometry_reject_fov_identically(fov):
    with pytest.raises(ValueError):
        yaml_dict_to_fov({"fov": fov})
    with pytest.raises(ValueError):
        SpatialGeometry.from_xy((4, 6), fov)


def test_config_and_geometry_reject_unitless_fov():
    with pytest.raises(u.UnitConversionError):
        yaml_dict_to_fov({"fov": [1, 2]})
    with pytest.raises(u.UnitConversionError):
        SpatialGeometry.from_xy((4, 6), [1, 2])
