import jax.numpy as jnp
import nifty.re as jft
from astropy import constants
from astropy import units as u
from jax import Array

from ...grid import Grid
from ...instruments.jwst.parametric_model.parametric_prior import (
    build_parametric_prior_from_prior_config,
)
from ...parse.sky_model.multifrequency.black_body import (
    BlackBodyConfig,
    ModifiedBlackBodyConfig,
)
from ..parametric_model.gaussian import Gaussian
from ..single_correlated_field import build_single_correlated_field_from_config
from .spectral_product_mf_sky import (
    SpectralProductSky,
    build_simple_spectral_sky_from_grid,
)


class BlackBody(jft.Model):
    def __init__(
        self,
        frequencies: u.Quantity,
        log_temperature: jft.Model,
        sky_unit: u.Unit,
    ) -> None:
        """
        Parameters
        ----------
        frequencies: u.Quantity
            The frequencies at which to evaluate the black body spectrum.
        log_temperature: jft.Model
            The statistical model for the temperature, assumed to have units of
            log(Kelvin).
        sky_unit: u.Unit
            The unit of the sky (spectral radiance).
        """
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


def build_black_body_spectrum_from_grid(
    grid: Grid,
    prefix: str,
    config: BlackBodyConfig,
    sky_unit: u.Unit,
    redshift: float = 0.0,
    spatial_unit: u.Unit = u.Unit("arcsec"),
) -> BlackBody:
    assert config.is_field

    centers = [u.Quantity(cen.value * cen.unit) for cen in grid.spectral.centers]
    frequencies = u.Quantity(centers).to(u.Unit("Hz"), equivalencies=u.spectral())
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
        log_gaussian = Gaussian.from_config(
            domain_key=f"{prefix}_temperature_gaussian",
            config=config.temperature_gaussian,
            coordinates=grid.spatial.get_xycoords(centered=True, unit=u.Unit("arcsec")),
            log=True,
        )
        log_temperature = jft.Model(
            lambda x: _log_temperature(x) + log_gaussian(x),
            domain=_log_temperature.domain | log_gaussian.domain,
        )

    return BlackBody(
        frequencies=frequencies,
        log_temperature=log_temperature,
        sky_unit=sky_unit,
    )


def build_black_body_spectrum(
    prefix: str,
    frequencies: u.Quantity,
    config: BlackBodyConfig,
    sky_unit: u.Unit,
    redshift: float = 0.0,
    spatial_unit: u.Unit = u.Unit("arcsec"),
) -> BlackBody:
    assert not config.is_field

    frequencies = frequencies.to(u.Unit("Hz"), equivalencies=u.spectral())
    frequencies = (1 + redshift) * frequencies

    log_temperature = build_parametric_prior_from_prior_config(
        domain_key=f"{prefix}_temperature",
        prior_config=config.temperature,
        shape=(),
        as_model=True,
    )
    return BlackBody(
        frequencies=frequencies,
        log_temperature=log_temperature,
        sky_unit=sky_unit,
    )


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
    """Build a ModifiedBlackBody from the grid and config.

    Parameters
    ----------
    grid: Grid
        The grid specifying the spatial extend, etc.
    prefix: str
        The prefix key to the model.
    config: dict | ModifiedBlackBodyConfig
        The parameters of the `ModifiedBlackBody`, see `ModifiedBlackBodyConfig`
        for settings.
    sky_unit: u.Unit
        The radiance unit.
    redshift: float
        The redshift of the source, this shifts the spectrum by (1+z).
    spatial_unit: u.Unit (default `arcsec`)
        Set the spatial units of the sky, these change the prior settings.
    spectral_unit: u.Unit (default `eV`)
        Set the spectral units of the sky, these change the prior settings.
    """
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
        log_gaussian = Gaussian.from_config(
            domain_key=f"{prefix}_temperature_gaussian",
            config=config.temperature_gaussian,
            coordinates=grid.spatial.get_xycoords(centered=True, unit=u.Unit("arcsec")),
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
