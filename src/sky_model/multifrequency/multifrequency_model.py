from typing import Union

from ...grid import Grid
from ...parse.parsing_base import FromYamlDict
from ...parse.sky_model.multifrequency.mf_model_from_grid import (
    SimpleSpectralSkyConfig,
    GreyBodyConfig,
    ConstantMFConfig,
)
from .mf_constant import SingleValueMf, build_constant_mf_from_grid
from .mf_model_from_grid import (
    build_grey_body_spectrum_from_grid,
    build_simple_spectral_sky_from_grid,
)
from .spectral_product_mf_sky import SpectralProductSky

# Direct mapping to config classes
MODEL_CONFIG_CLASSES: dict[str, FromYamlDict] = {
    "nifty_mf": SimpleSpectralSkyConfig,
    "niftymf": SimpleSpectralSkyConfig,
    "constant_mf": ConstantMFConfig,
    "grey_body": GreyBodyConfig,
}


def parsing_mf_model(
    model_cfg: dict,
    model_key: str,
) -> Union[FromYamlDict, SimpleSpectralSkyConfig, GreyBodyConfig, ConstantMFConfig]:
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
    grid: Grid, prefix: str, model_cfg: dict, **kwargs: dict
) -> SpectralProductSky | SingleValueMf:
    assert len(model_cfg) == 1
    model_key = next(iter(model_cfg.keys()))
    prefix = "_".join((prefix, model_key))

    # NOTE: This should be parsed before here:
    config = parsing_mf_model(model_cfg, model_key)

    if isinstance(config, SimpleSpectralSkyConfig):
        return build_simple_spectral_sky_from_grid(grid, prefix, config, **kwargs)

    elif isinstance(config, ConstantMFConfig):
        return build_constant_mf_from_grid(grid, prefix, config, **kwargs)

    elif isinstance(config, GreyBodyConfig):
        return build_grey_body_spectrum_from_grid(grid, prefix, config, **kwargs)

    else:
        raise ValueError(
            f"Invalid multifrequency model: {model_key}"
            f"\n Supported models: {set(MODEL_CONFIG_CLASSES.keys())}"
        )
