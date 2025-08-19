import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import constants, coordinates
from astropy import units as u
from jax import Array

from ...grid import Grid
from ...instruments.jwst.parametric_model.parametric_prior import (
    build_parametric_prior_from_prior_config,
)
from ...parse.sky_model.multifrequency.mf_model_from_grid import (
    ModifiedBlackBodyConfig,
    SimpleSpectralSkyConfig,
)
from ..parametric_model.gaussian import Gaussian
from ..single_correlated_field import build_single_correlated_field_from_config
from .spectral_product_mf_sky import SpectralProductSky, build_simple_spectral_sky


def build_simple_spectral_sky_from_grid(
    grid: Grid,
    prefix: str,
    config: SimpleSpectralSkyConfig | dict,
    spatial_unit: u.Unit = u.Unit("arcsec"),
    spectral_unit: u.Unit = u.Unit("eV"),
) -> SpectralProductSky:
    if isinstance(config, dict):
        config = SimpleSpectralSkyConfig.from_yaml_dict(config)

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


class ModifiedBlackBody(jft.Model):
    def __init__(
        self,
        black_body: BlackBody,
        optical_depth: SpectralProductSky,
    ) -> None:
        self.black_body = black_body
        self.optical_depth = optical_depth

        super().__init__(domain=optical_depth.domain | black_body.domain)

    def emissivity(self, x):
        return 1 - jnp.exp(-self.optical_depth(x))

    def __call__(self, x) -> Array:
        return self.emissivity(x) * self.black_body(x)


def build_modified_black_body_spectrum_from_grid(
    grid: Grid,
    prefix: str,
    config: ModifiedBlackBodyConfig,
    sky_unit: u.Unit,
    redshift: float = 0.0,
    spatial_unit: u.Unit = u.Unit("arcsec"),
    spectral_unit: u.Unit = u.Unit("eV"),
) -> ModifiedBlackBody:
    centers = [u.Quantity(cen.value * cen.unit) for cen in grid.spectral.centers]
    frequencies = u.Quantity(centers).to(u.Hz, equivalencies=u.spectral())
    frequencies = (1 + redshift) * frequencies

    _log_temperature = build_single_correlated_field_from_config(
        prefix=f"{prefix}_temperature",
        shape=grid.spatial.shape,
        distances=grid.spatial.distances.to(spatial_unit).value,
        config=config.temperature,
    )

    if not config.temperature_gaussian:
        log_temperature = _log_temperature
    else:
        gpre = f"{prefix}_temperature_gaussian"
        log_gaussian = Gaussian(
            i0=build_parametric_prior_from_prior_config(
                domain_key=f"{gpre}_i0",
                prior_config=config.temperature_gaussian.i0,
                shape=(),
                as_model=True,
            ),
            center=build_parametric_prior_from_prior_config(
                domain_key=f"{gpre}_center",
                prior_config=config.temperature_gaussian.center,
                shape=(2,),
                as_model=True,
            ),
            covariance=build_parametric_prior_from_prior_config(
                domain_key=f"{gpre}_covariance",
                prior_config=config.temperature_gaussian.covariance,
                shape=(2,),
                as_model=True,
            ),
            off_diagonal=build_parametric_prior_from_prior_config(
                domain_key=f"{gpre}_offdiagonal",
                prior_config=config.temperature_gaussian.off_diagonal,
                shape=(),
                as_model=True,
            ),
            coordiantes=grid.spatial.get_xycoords(centered=True, unit=u.Unit("arcsec")),
            log=True,
        )
        log_temperature = jft.Model(
            lambda x: _log_temperature(x) + log_gaussian(x),
            domain=_log_temperature.domain | log_gaussian.domain,
        )

    black_body = BlackBody(
        frequencies=frequencies,
        log_temperature=log_temperature,
        sky_unit=sky_unit,
    )

    optical_depth = build_simple_spectral_sky_from_grid(
        grid=grid,
        prefix=f"{prefix}_optical_depth",
        config=config.optical_depth,
        spatial_unit=spatial_unit,
        spectral_unit=spectral_unit,
    )

    return ModifiedBlackBody(
        optical_depth=optical_depth,
        black_body=black_body,
    )
