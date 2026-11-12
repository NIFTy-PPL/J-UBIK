"""The Gaussian likelihood of one observation given predicted visibilities."""

import re

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.likelihood import build_likelihood
from jubik.instruments.resolve.likelihood.mosaic_likelihood import (
    build_likelihood_from_sky_beamer,
)
from jubik.instruments.resolve.mosaicing.sky_beamer import BeamPattern, SkyBeamer
from jubik.instruments.resolve.noise.factory_noise_correction import (
    factory_noise_correction_model,
)
from jubik.instruments.resolve.parse.noise import BaseLineCorrection
from jubik.instruments.resolve.phase_shift import PhaseShift, _phase_per_arcsec
from jubik.likelihood import connect_likelihood_to_model
from jubik.utils import HoistedConstants

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)


def random_obs():
    pol = ju.polarization.PolarizationType(("LL", "RR"))
    return generate_random_obs(
        np.array([1e11, 1.1e11]), 30, [-1e3, 1e3], [-5, 5], pol,
        rng=np.random.default_rng(5),
    )


def test_weights_are_the_inverse_noise_covariance():
    obs = random_obs()
    d, w = obs.vis_val, obs.weight_val
    scaled = jft.Model(lambda s: s * d, domain=jft.ShapeWithDtype(()))  # bare-array input

    likelihood = build_likelihood(obs, scaled)

    assert_allclose(likelihood.energy(0.0), 0.5 * np.sum(w * np.abs(d) ** 2), rtol=1e-12)


def test_inverse_std_is_per_real_and_imaginary_part():
    obs = random_obs()
    d = obs.vis_val
    zero = jft.Model(lambda x: x["s"] * d, domain={"s": jft.ShapeWithDtype(())})
    inverse_std = jft.Model(lambda x: x["inv"] * np.ones(d.shape), domain={"inv": jft.ShapeWithDtype(())})

    likelihood = build_likelihood(obs, zero, inverse_std)

    energy = likelihood.energy(jft.Vector({"s": 0.0, "inv": 2.0}))
    assert_allclose(energy, 0.5 * 4 * np.sum(np.abs(d) ** 2) - 2 * d.size * np.log(2), rtol=1e-12)


def test_flagged_points_are_left_out():
    obs = random_obs()
    d, w = obs.vis_val.copy(), obs.weight_val.copy()
    d[0, 0, 0], w[0, 0, 0] = np.nan, 0.0  # a flagged point may hold NaN
    obs = rve.Observation(obs.antenna_positions, d, w, obs.legacy_polarization, obs.freq)
    good = w > 0
    vis = jft.Model(lambda x: x["s"] * np.nan_to_num(d), domain={"s": jft.ShapeWithDtype(())})
    sqrt_weight = jft.Model(lambda x: x["inv"] * np.sqrt(w), domain={"inv": jft.ShapeWithDtype(())})

    fixed = build_likelihood(obs, vis).energy(jft.Vector({"s": 0.0}))
    inferred = build_likelihood(obs, vis, sqrt_weight).energy(jft.Vector({"s": 0.0, "inv": 1.0}))

    expected = 0.5 * np.sum(w[good] * np.abs(d[good]) ** 2)
    assert_allclose(fixed, expected, rtol=1e-12)
    assert_allclose(inferred, expected - np.sum(np.log(w[good])), rtol=1e-12)


def test_likelihood_keeps_the_model_initializers():
    obs = random_obs()
    one = lambda key: np.ones(())
    # a callable and a structured initializer
    vis = jft.Model(lambda x: x["s"] * obs.vis_val, init=lambda key: {"s": np.ones(())})
    ones = np.ones(obs.vis.shape)
    inverse_std = jft.Model(lambda x: x["inv"] * ones, init=jft.Initializer({"inv": one}))

    for likelihood in (build_likelihood(obs, vis), build_likelihood(obs, vis, inverse_std)):
        start = likelihood.init(jax.random.PRNGKey(0))
        assert all(v == 1.0 for v in jax.tree.leaves(start))


def test_wrong_shape_raises():
    obs = random_obs()
    wrong = jft.Model(lambda x: x["v"], domain={"v": jft.ShapeWithDtype((1, 2, 3), complex)})
    with pytest.raises(ValueError, match="predicted visibilities"):
        build_likelihood(obs, wrong)

    vis = jft.Model(lambda x: x["v"], domain={"v": jft.ShapeWithDtype(obs.vis.shape, complex)})
    scalar = jft.Model(lambda x: x["inv"], domain={"inv": jft.ShapeWithDtype(())})
    with pytest.raises(ValueError, match="inverse_std"):
        build_likelihood(obs, vis, scalar)


@pytest.mark.parametrize("wrap", (dict, jft.Vector))
def test_hoisted_constants_are_leaves(wrap):
    big = np.arange(5000.0)
    model = jft.Model(lambda x: x["a"] * big, domain=wrap({"a": jft.ShapeWithDtype(())}))

    hoisted = HoistedConstants(model)

    assert any(np.shape(leaf) == big.shape for leaf in jax.tree.leaves(hoisted))
    # a superset of the domain, as a summed likelihood passes it
    assert_allclose(hoisted(wrap({"a": 2.0, "b": 1.0})), model({"a": 2.0}))


# a large constant: float, complex or integer, at least 400 elements (ducc keeps
# a fixed 317-byte blob)
LARGE_CONSTANT = re.compile(r"stablehlo\.constant.*?: tensor<((?:\d+x)+)(?:f|complex<f|i)\d")


@pytest.mark.parametrize(
    "settings",
    [
        rve.parse.FinufftSettings(epsilon=1e-6),
        rve.parse.Ducc0Settings(epsilon=1e-6, do_wgridding=False, nthreads=1, verbosity=False),
    ],
)
def test_compiled_likelihood_embeds_no_large_arrays(settings):
    """Every radio array reaches jit as an argument, not as a constant (#183)."""
    freqs = np.array([1.0e9, 1.5e9])
    pol = ju.polarization.PolarizationType(("I",))
    rng = np.random.default_rng(0)
    ant1, ant2 = rng.integers(0, 8, (2, 300))
    obs = generate_random_obs(
        freqs, 300, [-3e3, 3e3], [0, 0], pol, rng=rng, fraction_flagged=0.1,
        ant1=ant1, ant2=ant2, times=np.zeros(300),
    )
    spatial = ju.wcs.WcsAstropy(
        center=SkyCoord(ra=0 * u.deg, dec=0 * u.deg), shape=(32, 32),
        fov=u.Quantity((1.0, 1.0), u.deg),
    )
    grid = ju.Grid(
        spatial=spatial,
        spectral=ju.color.Color.from_central_frequencies(freqs),
        polarization=pol,
    )
    beam = rng.uniform(size=grid.shape)
    R = rve.interferometry_response(obs, grid, settings, beam=beam, sky_key="sky")
    shift = jft.Model(lambda x: x["shift"], domain={"shift": jft.ShapeWithDtype((2,))})
    phase = PhaseShift(shift, *_phase_per_arcsec(obs), obs.vis_val.dtype)
    gain = rng.uniform(size=obs.vis.shape)
    vis = jft.Model(lambda x: gain * phase(x) * R(x), domain=R.domain | phase.domain)
    noise = factory_noise_correction_model(BaseLineCorrection(alpha=2.0, scale=1.0), obs, "n")

    likelihood = build_likelihood(obs, vis) + build_likelihood(obs, vis, noise)
    sky = jft.Model(
        lambda x: {"sky": jnp.exp(x["xi"])}, domain={"xi": jft.ShapeWithDtype(grid.shape)}
    )
    likelihood = connect_likelihood_to_model(likelihood, sky)
    # the builder path sptLensing uses, without its sky beamer
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape), {"fld": BeamPattern(0.0, 0.0, beam, None)}
    )
    builders = [
        build_likelihood_from_sky_beamer(
            obs, "fld", sky_beamer, grid, settings, correction, shift=phase
        )
        for correction in (None, BaseLineCorrection(alpha=2.0, scale=1.0))
    ]
    old = builders[0].likelihood + builders[1].likelihood

    for likelihood in (likelihood, old):
        x = jft.zeros_like(likelihood.domain)
        # the likelihood as a jit argument, as nifty's optimize_kl passes it
        programs = (
            jax.jit(lambda lh, x: jax.value_and_grad(lh)(x)).lower(likelihood, x),
            jax.jit(lambda lh, x: lh.metric(x, x)).lower(likelihood, x),
        )
        for program in programs:
            sizes = [
                np.prod([int(n) for n in m.group(1).split("x") if n])
                for m in LARGE_CONSTANT.finditer(program.as_text())
            ]
            assert [s for s in sizes if s >= 400] == []
