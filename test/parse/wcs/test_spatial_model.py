from jubik.parse.wcs.spatial_model import SpatialModel, resolve_str_to_quantity
from jubik.parse.wcs.coordinate_system import CoordinateSystems

from astropy.coordinates import SkyCoord
import astropy.units as u
import pytest


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


def test_legacy_spatial_keys_are_rejected():
    with pytest.raises(ValueError, match="use `shape`"):
        SpatialModel.from_yaml_dict({"sdim": 8, "fov": "1arcsec"})

    with pytest.raises(ValueError, match="use astronomical `position_angle`"):
        SpatialModel.from_yaml_dict(
            {"shape": 8, "fov": "1arcsec", "rotation": "1deg"}
        )


def test_resolve_str_to_quantity():
    quantities = {
        "12.3muas": 12.3 * u.microarcsecond,
        "12.3mas": 12.3 * u.milliarcsecond,
        "12.3as": 12.3 * u.arcsecond,
        "12.3amin": 12.3 * u.arcmin,
        "12.3deg": 12.3 * u.deg,
        "12.3rad": 12.3 * u.rad,
    }

    for key, val in quantities.items():
        assert resolve_str_to_quantity(key) == val
