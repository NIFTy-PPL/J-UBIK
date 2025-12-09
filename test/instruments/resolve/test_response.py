import jax
import jax, numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
import jubik0 as ju
import jubik0.instruments.resolve as rve

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


def prepare_grid_obs_response(pol_sky, pol_channels, freqs):
    pol_type_sky = ju.polarization.PolarizationType(pol_sky)
    pol_type_data = ju.polarization.PolarizationType(pol_channels)
    obs = generate_random_obs(
        freqs, 50, [-1e2, 1e2], [-5, 5], pol_type_data, fraction_flagged=0.3
    )
    sky_center = SkyCoord(ra=np.nan * u.Unit("rad"), dec=np.nan * u.Unit("rad"))
    fov = u.Quantity((u.Quantity("1deg"), u.Quantity("1.5deg")))
    spacial = ju.wcs.WcsAstropy(center=sky_center, shape=(100, 100), fov=fov)
    spectral = ju.color.ColorRanges.from_freqs(freqs)
    grid = ju.Grid(spatial=spacial, spectral=spectral, polarization=pol_type_sky)

    ducc0_settings = rve.parse.Ducc0Settings(
        epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
    )
    finufft_settings = rve.parse.FinufftSettings(epsilon=1e-9)
    r_ducc = rve.interferometry_response(obs, grid, backend_settings=ducc0_settings)
    r_finufft = rve.interferometry_response(
        obs, grid, backend_settings=finufft_settings
    )
    return grid, obs, r_ducc, r_finufft


@pmp("pol_sky", (("I",), ("I", "Q", "U", "V")))
@pmp("pol_channels", (("LL", "RR"), ("RR", "RL", "LR", "LL"), ("XX", "XY", "YX", "YY")))
@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
def test_response_ducc_finufft_consistency(pol_sky, pol_channels, freqs):
    if pol_sky == ("I", "Q", "U", "V") and pol_channels == ("LL", "RR"):
        pytest.skip("Only stokes I data.")
    if pol_sky == ("I",) and not pol_channels == ("LL", "RR"):
        pytest.skip("Only stokes I sky.")
    grid, obs, r_ducc, r_finufft = prepare_grid_obs_response(
        pol_sky, pol_channels, freqs
    )

    sky = np.random.normal(size=grid.shape)
    res_ducc = r_ducc(sky)
    res_finufft = r_finufft(sky)

    assert_allclose(res_ducc, res_finufft, rtol=1e-9, atol=1e-9)


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
def test_response_StokesI(freqs):
    pol_sky = ("I",)
    pol_channels = ("LL", "RR")
    grid, obs, r_ducc, r_finufft = prepare_grid_obs_response(
        pol_sky, pol_channels, freqs
    )

    sky = jnp.zeros(grid.shape)
    cx = grid.shape[3] // 2
    cy = grid.shape[4] // 2
    dvol = grid.spatial.dvol.to(u.rad**2).value
    sky[:, :, :, cx, cy] = 1 / dvol
    vis = r_ducc(sky)
    vis_expected = np.ones_like(obs.vis, dtype=jnp.float64)
    assert_allclose(vis, vis_expected)


if __name__ == "__main__":
    # test_response_ducc_finufft_consistency(
    #     ("I", "Q", "U", "V"), ("RR", "RL", "LR", "LL"), np.array([1e9, 1.3e9, 2e9])
    # )
    # test_response_ducc_finufft_consistency(
    #     ("I", "Q", "U", "V"), ("XX", "XY", "YX", "YY"), np.array([1e9, 1.3e9, 2e9])
    # )
    test_response_StokesI(np.array([1e9, 1.3e9, 2e9]))
