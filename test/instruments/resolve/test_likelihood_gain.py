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
from jubik.instruments.resolve.likelihood.mosaic_likelihood import (
    VariableLikelihoodBuilder,
    build_likelihood_from_sky_beamer,
)
from jubik.instruments.resolve.likelihood.radio_likelihood import (
    RadioLikelihoodProducts,
)
from jubik.instruments.resolve.mosaicing.sky_beamer import BeamPattern, SkyBeamer
from jubik.instruments.resolve.parse.noise import BaseLineCorrection

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

FREQS = np.array([1e9, 1.3e9])
POL = ju.polarization.PolarizationType(("LL", "RR"))
FIELD = "band7_fld"
N_ROWS = 40


def sky_grid():
    center = SkyCoord(ra=np.nan * u.rad, dec=np.nan * u.rad)
    spatial = ju.wcs.WcsAstropy(center=center, shape=(32, 32), fov=[1.0, 1.5] * u.deg)
    return ju.Grid(
        spatial=spatial,
        spectral=ju.color.Color.from_central_frequencies(FREQS),
        polarization=ju.polarization.PolarizationType(("I",)),
    )


def sky_beamer():
    shape = sky_grid().shape
    return SkyBeamer(
        jft.ShapeWithDtype(shape), {FIELD: BeamPattern(0.0, 0.0, np.ones(shape), None)}
    )


def band7(backend="finufft", noise=None):
    """Builder of one random observation as one field, without a gain."""
    np.random.seed(45)
    # the variable noise model works per baseline, so the rows need antennas
    obs = generate_random_obs(
        FREQS,
        N_ROWS,
        [-1e2, 1e2],
        [-5, 5],
        POL,
        ant1=np.random.choice([0, 2, 5], N_ROWS),
        ant2=np.random.choice([1, 3], N_ROWS),
        times=np.random.uniform(0, 80, N_ROWS),
    )
    grid = sky_grid()
    settings = {
        "ducc": rve.parse.Ducc0Settings(
            epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
        ),
        "finufft": rve.parse.FinufftSettings(epsilon=1e-9),
    }[backend]
    return build_likelihood_from_sky_beamer(
        observation=obs,
        field_name=FIELD,
        sky_beamer=sky_beamer(),
        sky_grid=grid,
        backend_settings=settings,
        noise_std_correction=noise,
    )


# Any model of the visibility factor will do: one free complex gain per row.
GAIN = jft.Model(
    lambda x: jnp.exp((0.1 + 1j) * x["gain"]),
    domain={"gain": jft.ShapeWithDtype((2, N_ROWS, 1))},
)


def random_latent(domain, seed=0):
    return jft.random_like(jax.random.PRNGKey(seed), domain)


def random_sky(seed=2):
    return {FIELD: np.random.default_rng(seed).normal(size=sky_grid().shape)}


def test_with_gain_multiplies_the_gain_onto_a_copy():
    plain = band7()
    calibrated = plain.with_gain(GAIN)
    sky = random_sky()
    x = random_latent(GAIN.domain)

    assert plain.gain is None and calibrated.gain is GAIN
    assert type(calibrated) is type(plain)
    assert_allclose(
        calibrated.response(sky | x), GAIN(x) * plain.response(sky), rtol=1e-12
    )
    assert_allclose(
        calibrated.likelihood.energy(jft.Vector(sky | jft.zeros_like(GAIN.domain))),
        plain.likelihood.energy(jft.Vector(sky)),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="already has a gain"):
        calibrated.with_gain(GAIN)


@pytest.mark.parametrize(
    "noise", [None, BaseLineCorrection(alpha=2.0, scale=1.0)], ids=["fixed", "variable"]
)
def test_calibration_likelihood_is_the_likelihood_at_fixed_sky(noise):
    plain = band7(noise=noise)
    calibrated = plain.with_gain(GAIN)
    sky = random_sky()

    likelihood = calibrated.calibration_likelihood(plain.response(sky))
    x = random_latent(likelihood.domain)

    assert isinstance(calibrated, VariableLikelihoodBuilder) == (noise is not None)
    assert likelihood.domain.tree.keys() == calibrated.likelihood.domain.tree.keys() - {
        FIELD
    }
    assert_allclose(
        likelihood.energy(x),
        calibrated.likelihood.energy(jft.Vector(sky | x.tree)),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="gain"):
        plain.calibration_likelihood(plain.response(sky))


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_response_adjoint_holds_the_gain_fixed(backend):
    """<R_x s, d> == <s, R_x^H d> with the gain fixed at a sample x."""
    plain = band7(backend)
    calibrated = plain.with_gain(GAIN)
    sky = random_sky()
    x = random_latent(GAIN.domain)
    d = calibrated.visibilities

    adjoint = calibrated.response_adjoint(d, x)[FIELD]

    lhs = np.real(np.sum(np.conj(calibrated.response(sky | x)) * d))
    assert_allclose(lhs, np.sum(sky[FIELD] * np.real(adjoint)), rtol=1e-6)
    # The gain reaches the adjoint, so it differs from the one without gain.
    uncalibrated = plain.response_adjoint(d)[FIELD]
    assert np.linalg.norm(adjoint - uncalibrated) > 0.1 * np.linalg.norm(uncalibrated)
    with pytest.raises(ValueError, match="gain"):
        calibrated.response_adjoint(d)


def test_products_with_gain_gives_every_builder_its_gain():
    plain = band7()
    beamer = sky_beamer()
    products = RadioLikelihoodProducts(
        [plain, plain],
        sky_beamer=jft.Model(lambda x: beamer(x["sky"]), domain={"sky": beamer.domain}),
        radio_sky_extractor=None,
    )
    shape = GAIN.domain["gain"]
    other = jft.Model(lambda x: jnp.exp(1j * x["other"]), domain={"other": shape})

    calibrated = products.with_gain([GAIN, other])

    assert [b.gain for b in calibrated.likelihoods] == [GAIN, other]
    assert [b.gain for b in products.likelihoods] == [None, None]
    sky = {"sky": random_sky()[FIELD]}
    zero_gains = {"gain": jnp.zeros(shape.shape), "other": jnp.zeros(shape.shape)}
    assert calibrated.likelihood.domain.tree.keys() == {"sky", "gain", "other"}
    assert_allclose(
        calibrated.likelihood.energy(jft.Vector(sky | zero_gains)),
        products.likelihood.energy(jft.Vector(sky)),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="one gain per likelihood"):
        products.with_gain([GAIN])
