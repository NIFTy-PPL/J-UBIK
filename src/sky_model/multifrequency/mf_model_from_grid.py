from typing import Callable

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import constants
from astropy import units as u
from jax import Array
from nifty.re.correlated_field import (
    MaternAmplitude,
    NonParametricAmplitude,
    hartley,
    make_grid,
)
from nifty.re.num.stats_distributions import lognormal_prior, normal_prior

from ...grid import Grid
from ...parse.sky_model.multifrequency.mf_model_from_grid import (
    GreyBodyConfig,
    SimpleSpectralSkyConfig,
)
from ..single_correlated_field import build_single_correlated_field
from .spectral_product_mf_sky import SpectralProductSky, build_simple_spectral_sky
from .spectral_product_utils.distribution_or_default import (
    build_distribution_or_default,
)
from .spectral_product_utils.frequency_deviations import (
    build_frequency_deviations_model_with_degeneracies,
)
from .spectral_product_utils.normalized_amplitude_model import (
    assert_normalized_amplitude_model,
    build_normalized_amplitude_model,
)
from .spectral_product_utils.scaled_excitations import (
    ScaledExcitations,
    build_scaled_excitations,
)


def build_simple_spectral_sky_from_grid(
    grid: Grid,
    prefix: str,
    config: SimpleSpectralSkyConfig,
    spatial_unit: u.Unit = u.Unit("arcsec"),
    spectral_unit: u.Unit = u.Unit("eV"),
) -> SpectralProductSky:
    # NOTE: Spatial settings
    shape = grid.spatial.shape
    distances = grid.spatial.distances.to(spatial_unit).value

    # NOTE: Spectral settings
    ref_energy = config.reference_bin
    log_energies = np.log(
        [c.to_unit(spectral_unit).value for c in grid.spectral.centers]
    )

    return build_simple_spectral_sky(
        prefix=prefix,
        shape=shape,
        distances=distances,
        log_frequencies=log_energies,
        reference_frequency_index=ref_energy,
        zero_mode_settings=config.zero_mode,
        spatial_amplitude_settings=config.spatial_amplitude,
        spectral_index_settings=config.spectral_index,
        spectral_amplitude_settings=config.spectral_amplitude,
        deviations_settings=config.spectral_deviations,
        spatial_amplitude_model=config.spatial_amplitude_model,
        spectral_amplitude_model=config.spectral_amplitude_model,
        nonlinearity=config.nonlinearity,
    )


class BlackBody(jft.Model):
    def __init__(
        self,
        frequencies: u.Quantity,
        log_temperature: jft.Model,
        sky_unit: u.Unit,
    ) -> None:
        # NOTE : Always in Kelvin

        outshape = (Ellipsis,) + (None,) * len(log_temperature.target.shape)

        self._hv_k = (
            ((constants.h / constants.k_B) * frequencies).to(u.Unit("K")).value
        )[outshape]

        # NOTE : Prefactor of Planck's law (in sky_energy_unit)

        self._2hvvv_cc = (
            (2 * constants.h * frequencies**3 / constants.c**2 / (1 * u.Unit("sr")))
            .to(sky_unit, equivalencies=u.spectral())
            .value
        )[outshape]

        # TODO : Make beta more efficient by changing the prior initialization from
        # log_temperature to beta, and have log_temperature be a method.
        self.log_temperature = log_temperature

        super().__init__(domain=log_temperature.domain)

    def beta(self, x):
        # TODO : Make beta more efficient by changing the prior initialization from
        # log_temperature to beta, and have log_temperature be a method.
        return jnp.exp(-self.log_temperature(x))

    def temperature(self, x):
        return jnp.exp(self.log_temperature(x))

    def __call__(self, x) -> Array:
        hv_kT = self.beta(x) * self._hv_k
        return self._2hvvv_cc / (jnp.exp(hv_kT) - 1.0)


class GreyBody(jft.Model):
    def __init__(
        self, black_body: BlackBody, emissivity_tau: SpectralProductSky
    ) -> None:
        self.black_body = black_body
        self.emissivity_tau = emissivity_tau

        super().__init__(domain=black_body.domain | emissivity_tau.domain)

    def emissivity(self, x):
        return 1 - jnp.exp(-self.emissivity_tau(x))

    def __call__(self, x) -> Array:
        return self.emissivity(x) * self.black_body(x)


def build_grey_body_spectrum_from_grid(
    grid: Grid,
    prefix: str,
    config: GreyBodyConfig,
    sky_unit: u.Unit,
    spatial_unit: u.Unit = u.Unit("arcsec"),
    spectral_unit: u.Unit = u.Unit("eV"),
) -> jft.Model:
    log_temperature, _ = build_single_correlated_field(
        prefix=f"{prefix}_temperature",
        shape=grid.spatial.shape,
        distances=grid.spatial.distances.to(spatial_unit).value,
        zero_mode_config=config.temperature_zero_mode,
        fluctuations_config=config.temperature_fluctuations,
    )

    black_body = BlackBody(
        u.Quantity(grid.spectral.centers).to(u.Hz, equivalencies=u.spectral()),
        log_temperature=log_temperature,
        sky_unit=sky_unit,
    )

    emissivity_tau = build_simple_spectral_sky_from_grid(
        grid=grid,
        prefix=f"{prefix}_emissivity",
        config=config.emissivity,
        spatial_unit=spatial_unit,
        spectral_unit=spectral_unit,
    )
    return GreyBody(black_body=black_body, emissivity_tau=emissivity_tau)
