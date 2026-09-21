"""Self-contained regression checks for geometry in data-dependent demos."""

import ast
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

import jubik as ju
from jubik.parse.wcs.spatial_model import yaml_dict_to_square_size
from jubik.parse.grid import GridModel


ROOT = Path(__file__).resolve().parents[1]


def _assignment(path, name):
    tree = ast.parse(path.read_text())
    return next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


def _evaluate(expression, namespace):
    # Execute the actual demo expression, not its data/PSF-dependent pipeline.
    return eval(compile(ast.Expression(expression), "demo geometry", "eval"), namespace)


@pytest.mark.parametrize("relative", ["demos/jwst_demo.py", "docs/source/user/jwst_demo.py"])
def test_jwst_demo_constructs_reconstruction_and_data_grids(relative):
    path = ROOT / relative
    namespace = dict(
        ju=ju, u=u, SkyCoord=SkyCoord, pointing_center=(0.2, -0.1),
        cfg={"grid": {"shape": (16, 8), "fov": 6}},
        data_shape=12, data_fov=4,
    )
    reconstruction = _evaluate(_assignment(path, "reconstruction_grid"), namespace)
    data = _evaluate(_assignment(path, "data_grid"), namespace)
    assert reconstruction.spatial.shape_yx == (8, 16)
    assert data.spatial.shape_yx == (12, 12)
    expected = SkyCoord(0.2 * u.rad, -0.1 * u.rad)
    assert reconstruction.spatial.center.separation(expected) < 1e-10 * u.arcsec
    assert data.spatial.center.separation(expected) < 1e-10 * u.arcsec


def test_grid_shape_remains_the_full_numerical_shape():
    grid = ju.Grid.from_shape_and_fov(
        shape=(16, 8), fov=(16, 8) * u.arcsec
    )
    assert grid.shape == (1, 1, 1, 8, 16)
    assert grid.array_shape == grid.shape


@pytest.mark.parametrize("shape", [16, (16, 16), (16, 8)])
def test_psf_demo_checks_square_shape_before_building_sky(shape):
    path = ROOT / "demos/test_psf.py"
    expression = _assignment(path, "spix")
    namespace = dict(cfg={"grid": {"shape": shape}},
                     yaml_dict_to_square_size=yaml_dict_to_square_size)
    if shape == (16, 8):
        with pytest.raises(ValueError, match="eROSITA PSF demo currently requires a square grid"):
            _evaluate(expression, namespace)
    else:
        assert _evaluate(expression, namespace) == 16
    tree = ast.parse(path.read_text())
    assignments = {t.id: n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert assignments["spix"] < assignments["sky"]


def test_grid_demo_draws_same_yx_sky_on_rotated_celestial_axes():
    demo = runpy.run_path(str(ROOT / "demos/grids.py"))
    fig = demo["plot_sky_coordinates"]()
    try:
        axes = fig.axes[:2]
        np.testing.assert_array_equal(axes[0].images[0].get_array(), axes[1].images[0].get_array())
        assert axes[0].images[0].get_array().shape == (64, 96)
        for ax in axes:
            assert ax.images[0].origin == "lower"
            assert ax.wcs.wcs.radesys == "ICRS"
        np.testing.assert_allclose(axes[0].wcs.wcs.pc, np.eye(2))
        assert not np.allclose(axes[1].wcs.wcs.pc, np.eye(2))
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("frame,equinox", [("icrs", None), ("fk5", "J1990.0"), ("fk4", "B1950.0")])
def test_documented_equatorial_config_reaches_wcs_metadata(frame, equinox):
    config = dict(shape=[16, 8], fov=["48arcsec", "24arcsec"],
                  frame=frame, sky_center=dict(ra="10deg", dec="-5deg"),
                  position_angle="30deg", energy_unit="eV",
                  energy_bin=dict(e_min=[1], e_max=[2], reference_bin=0))
    if equinox is not None:
        config["equinox"] = equinox
    grid = ju.Grid.from_grid_model(GridModel.from_yaml_dict(config))
    assert grid.spatial.shape_yx == (8, 16)
    assert grid.spatial.center.frame.name == frame
    assert grid.spatial.wcs.radesys == frame.upper()
    if equinox is not None:
        assert grid.spatial.wcs.equinox == float(equinox[1:])
    assert grid.spatial.position_angle == 30 * u.deg
