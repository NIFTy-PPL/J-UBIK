from jubik0.parse.wcs.spatial_model import SpatialModel, resolve_str_to_quantity
from jubik0.parse.wcs.coordinate_system import CoordinateSystems

from astropy.coordinates import SkyCoord
import astropy.units as u


def test_spatial_model_from_yaml_dict():
    grid_config = dict(
        sdim=384,
        fov="6arcsec",
        rotation="12.0deg",
        coordinate_frame="icrs",
        sky_center=dict(
            ra="64.66543063107049deg",
            dec="-47.86462563973049deg",
        ),
    )

    spatial_model = SpatialModel.from_yaml_dict(grid_config)

    scc = grid_config["sky_center"]
    assert spatial_model.wcs_model.center == SkyCoord(ra=scc["ra"], dec=scc["dec"])
    assert spatial_model.shape == (grid_config["sdim"],) * 2
    assert spatial_model.fov == (u.Quantity(grid_config["fov"]),) * 2
    assert spatial_model.wcs_model.rotation == u.Quantity(grid_config["rotation"])
    assert spatial_model.wcs_model.coordinate_system == CoordinateSystems.icrs.value


def test_spatial_model_from_config_parser():
    grid_config = {
        "image center ra": "64.66543063107049deg",
        "image center dec": "-47.86462563973049deg",
        "frame": "icrs",
        "space npix x": 384,
        "space npix y": 256,
        "space fov x": "12arcsec",
        "space fov y": "8arcsec",
        "space rotation": "8arcsec",
    }

    spatial_model = SpatialModel.from_config_parser(grid_config)

    # Test center
    center = SkyCoord(
        ra=u.Quantity(grid_config["image center ra"]),
        dec=u.Quantity(grid_config["image center dec"]),
        frame=grid_config["frame"],
    )
    assert spatial_model.wcs_model.center == center

    # Test shape
    assert spatial_model.shape == (
        grid_config["space npix x"],
        grid_config["space npix y"],
    )

    # Test rotation
    rotation = u.Quantity(grid_config["space rotation"])
    assert spatial_model.wcs_model.rotation == rotation

    # Test CoordinateSystem
    assert spatial_model.wcs_model.coordinate_system == CoordinateSystems.icrs.value


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
