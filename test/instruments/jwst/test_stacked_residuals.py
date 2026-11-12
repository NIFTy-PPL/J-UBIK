"""stacked_residuals: a residual shared by two frames peaks in the stack; pure noise has S/N std 1."""

import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.jwst.jwst_likelihoods import TargetLikelihoodProducts
from jubik.instruments.jwst.likelihood.likelihood import (
    GaussianLikelihoodBuilder,
    ShotNoise,
    ShotNoiseGaussianLikelihoodBuilder,
)
from jubik.instruments.jwst.likelihood.target_likelihood import SingleTargetLikelihood
from jubik.instruments.jwst.plotting.residuals import ResidualPlottingInformation
from jubik.instruments.jwst.plotting.stacked_residuals import stacked_residuals
from jubik.sky_filter import SKY_KEY, FilterWeights, SkyFilter
from jubik.wcs.wcs_astropy import WcsAstropy

NY = 16
GAIN = 2.0
RNG = np.random.default_rng(0)


def _products(data: dict, std: float, masks: dict, noise: ShotNoise | None = None) -> TargetLikelihoodProducts:
    """Frames whose response is GAIN times the band's sky plus a zero flux, on their mask; shot noise if `noise`."""
    spatial = WcsAstropy(
        center=SkyCoord(ra=64.665 * u.deg, dec=-47.865 * u.deg),
        shape=(NY, NY),
        fov=(1 * u.arcsec, 1 * u.arcsec),
    )
    grid = Grid(spatial=spatial, spectral=Color(np.array([1.0, 2.0]) * u.um))
    band = FilterWeights(slice(0, 1), np.ones(1))
    singles = []
    for name, d in data.items():
        mask = masks[name]
        response = jft.Model(
            lambda x, name=name, mask=mask: (GAIN * x[name] + x[f"{name}_zero_flux"])[mask],
            domain={
                name: jft.ShapeWithDtype((NY, NY), float),
                f"{name}_zero_flux": jft.ShapeWithDtype((), float),
            },
        )
        builder = (
            GaussianLikelihoodBuilder(response, d, np.full((NY, NY), std), mask)
            if noise is None
            else ShotNoiseGaussianLikelihoodBuilder(response, d, noise, mask)
        )
        singles.append(SingleTargetLikelihood(name, builder))
    return TargetLikelihoodProducts(
        likelihoods=singles,
        plotting=ResidualPlottingInformation(),
        sky_filter=SkyFilter(grid, {name: band for name in data}, dtype=float),
    )


SKY = jft.Model(
    lambda x: {SKY_KEY: x["s"][None]}, domain={"s": jft.ShapeWithDtype((NY, NY), float)}
)


def test_a_shared_residual_peaks_in_the_stack():
    peak = (3, 11)
    data, masks = {}, {}
    for name, zero_flux in (("a", 0.5), ("b", -0.3)):
        d = np.full((NY, NY), zero_flux)
        d[peak] += GAIN * 1.0  # a point the model sky lacks
        data[name] = d + 0.05 * RNG.standard_normal((NY, NY))
        masks[name] = RNG.random((NY, NY)) > 0.3
        masks[name][peak] = True
    products = _products(data, 0.05, masks)
    positions = [
        {"s": np.zeros((NY, NY)), "a_zero_flux": 0.5, "b_zero_flux": -0.3},
        jft.Vector({"s": np.zeros((NY, NY)), "a_zero_flux": 0.5, "b_zero_flux": -0.3}),
    ]
    out = stacked_residuals(products, SKY, positions, bands={"ab": ["a", "b"]})
    assert list(out) == ["ab"]
    # the stack is the mean of the samples' stacks, each at its own nuisances
    other = {"s": np.zeros((NY, NY)), "a_zero_flux": 0.1, "b_zero_flux": 0.2}
    mixed = stacked_residuals(products, SKY, [positions[0], other], bands={"ab": ["a", "b"]})
    alone = stacked_residuals(products, SKY, [other], bands={"ab": ["a", "b"]})
    np.testing.assert_allclose(mixed["ab"].stack, (out["ab"].stack + alone["ab"].stack) / 2, rtol=1e-10)
    snr = out["ab"].snr
    assert np.unravel_index(np.nanargmax(snr), snr.shape) == peak
    assert np.all(np.isnan(snr[~(masks["a"] | masks["b"])]))
    # the stack is the sum of GAIN (d - R(s)) / sigma^2 over the frames
    expected = sum(GAIN * (data[n][peak] - (0.5 if n == "a" else -0.3)) / 0.05**2 for n in "ab")
    np.testing.assert_allclose(out["ab"].stack[peak], expected, rtol=1e-10)


@pytest.mark.parametrize("shot", [False, True])
def test_pure_noise_has_unit_snr(shot):
    """Data drawn about the model at the position, fixed σ or σ² = a + b R(s)."""
    sky = 1.0 + RNG.random((NY, NY))
    noise = ShotNoise(a=0.01, b=0.05) if shot else None
    std = np.sqrt(0.01 + 0.05 * GAIN * sky) if shot else 0.2
    data = {name: GAIN * sky + std * RNG.standard_normal((NY, NY)) for name in "ab"}
    masks = {name: np.ones((NY, NY), bool) for name in "ab"}
    position = {"s": sky, "a_zero_flux": 0.0, "b_zero_flux": 0.0}
    out = stacked_residuals(_products(data, 0.2, masks, noise), SKY, [position], n_noise=400)
    for result in out.values():  # one band per frame by default
        assert 0.85 < result.snr.std() < 1.15
        assert abs(result.snr.mean()) < 0.2
