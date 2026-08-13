import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from numpy.testing import assert_allclose

from jubik.sky_model.stokes_adder import StokesAdder


pytestmark = pytest.mark.filterwarnings("ignore:drawing white parameters:UserWarning")


def build_stokes_adder(shape):
    domain = {"pre_stokes": jft.ShapeWithDtype((4,) + shape)}
    fields = (
        jft.Model(lambda x, ii=ii: x["pre_stokes"][ii], domain=domain)
        for ii in range(4)
    )
    return StokesAdder(fields)


@pytest.mark.parametrize("n_fields", (0, 1, 3, 5))
def test_stokes_adder_requires_four_fields(n_fields):
    with pytest.raises(ValueError, match="Expected four pre-Stokes field models"):
        StokesAdder([None] * n_fields)


def test_stokes_adder_matches_analytic_transformation():
    latent = jnp.array(
        [
            [[0.1, -0.2], [0.3, -0.4]],
            [[0.2, 0.3], [-0.1, 0.4]],
            [[-0.3, 0.1], [0.2, 0.2]],
            [[0.4, -0.2], [0.3, -0.1]],
        ]
    )
    model = build_stokes_adder(latent.shape[1:])

    result = model({"pre_stokes": latent})

    pol_int = np.sqrt(np.sum(np.asarray(latent[1:]) ** 2, axis=0))
    expected = np.concatenate(
        [
            np.exp(latent[:1]) * np.cosh(pol_int),
            np.exp(latent[:1])
            * (np.sinh(pol_int) / pol_int)[None]
            * latent[1:],
        ]
    )
    assert_allclose(result, expected)


def test_unpolarized_field_is_finite_and_has_zero_quv():
    latent = jnp.zeros((4, 3))
    latent = latent.at[0].set(jnp.array([-2.0, 0.0, 2.0]))
    model = build_stokes_adder(latent.shape[1:])

    result = jax.jit(lambda xx: model({"pre_stokes": xx}))(latent)

    assert np.all(np.isfinite(result))
    assert_allclose(result[0], np.exp(latent[0]))
    assert_allclose(result[1:], 0.0)


def test_unpolarized_field_has_finite_analytic_jacobian():
    latent = jnp.array([0.7, 0.0, 0.0, 0.0])
    model = build_stokes_adder(())

    jacobian = jax.jacfwd(lambda xx: model({"pre_stokes": xx}))(latent)

    assert np.all(np.isfinite(jacobian))
    assert_allclose(jacobian, np.exp(latent[0]) * np.eye(4), rtol=1e-6)


def test_stokes_adder_satisfies_polarization_constraint():
    latent = jax.random.normal(jax.random.PRNGKey(42), (4, 5, 3))
    model = build_stokes_adder(latent.shape[1:])

    result = model({"pre_stokes": latent})
    invariant = result[0] ** 2 - jnp.sum(result[1:] ** 2, axis=0)

    assert np.all(result[0] > 0)
    assert_allclose(invariant, jnp.exp(2 * latent[0]), rtol=1e-5, atol=1e-6)
