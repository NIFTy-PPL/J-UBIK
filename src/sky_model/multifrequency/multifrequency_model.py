from typing import Union

from astropy import units as u

from ...grid import Grid
from ...parse.parsing_base import FromYamlDict
from ...parse.sky_model.multifrequency.mf_model_from_grid import (
    ConstantMFConfig,
    ModifiedBlackBodyConfig,
    SimpleSpectralSkyConfig,
)
from .mf_constant import SingleValueMf, build_constant_mf_from_grid
from .mf_model_from_grid import (
    build_modified_black_body_spectrum_from_grid,
    build_simple_spectral_sky_from_grid,
)
from .spectral_product_mf_sky import SpectralProductSky

# Direct mapping to config classes
MODEL_CONFIG_CLASSES: dict[str, FromYamlDict] = {
    "nifty_mf": SimpleSpectralSkyConfig,
    "niftymf": SimpleSpectralSkyConfig,
    "constant_mf": ConstantMFConfig,
    "modified_black_body": ModifiedBlackBodyConfig,
}


def parsing_mf_model(
    model_cfg: dict,
    model_key: str,
) -> Union[
    FromYamlDict, SimpleSpectralSkyConfig, ModifiedBlackBodyConfig, ConstantMFConfig
]:
    """Parse multifrequency model using simple class mapping."""

    try:
        config_class = MODEL_CONFIG_CLASSES[model_key]
    except KeyError:
        raise ValueError(
            f"Invalid multifrequency model: {model_key}"
            f"\n Supported models: {set(MODEL_CONFIG_CLASSES.keys())}"
        )

    return config_class.from_yaml_dict(model_cfg[model_key])


def build_multifrequency_from_grid(
    grid: Grid,
    prefix: str,
    model_cfg: dict,
    spatial_unit: u.Unit = u.Unit("arcsec"),
    spectral_unit: u.Unit = u.Unit("eV"),
    redshift: float = 0.0,
    sky_unit: u.Unit = u.Unit("mJy") / u.Unit("arcsec"),
) -> SpectralProductSky | SingleValueMf:
    assert len(model_cfg) == 1
    model_key = next(iter(model_cfg.keys()))
    prefix = "_".join((prefix, model_key))

    # NOTE: This should be parsed before here:
    config = parsing_mf_model(model_cfg, model_key)

    if isinstance(config, ConstantMFConfig):
        return build_constant_mf_from_grid(grid, prefix, config)

    elif isinstance(config, SimpleSpectralSkyConfig):
        return build_simple_spectral_sky_from_grid(
            grid, prefix, config, spatial_unit=spatial_unit, spectral_unit=spectral_unit
        )

    elif isinstance(config, ModifiedBlackBodyConfig):
        return build_modified_black_body_spectrum_from_grid(
            grid,
            prefix,
            config,
            sky_unit,
            redshift=redshift,
            spatial_unit=spatial_unit,
            spectral_unit=spectral_unit,
        )

    else:
        raise ValueError(
            f"Invalid multifrequency model: {model_key}"
            f"\n Supported models: {set(MODEL_CONFIG_CLASSES.keys())}"
        )
