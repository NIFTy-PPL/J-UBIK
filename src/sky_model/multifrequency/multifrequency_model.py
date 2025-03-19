from ...grid import Grid

from .mf_model_from_grid import build_mf_model_from_grid
from .mf_constant import build_constant_mf_from_grid

NIFTY_MF = {"nifty_mf", "niftymf"}
CONSTANT = {"constant_mf"}

SUPPORTED_MODELS = CONSTANT | NIFTY_MF


def build_multifrequency_from_grid(
    grid: Grid, prefix: str, model_cfg: dict, **kwargs: dict
):

    assert len(model_cfg) == 1
    model_key = next(iter(model_cfg.keys()))
    prefix = "_".join((prefix, model_key))

    if model_key in NIFTY_MF:
        return build_mf_model_from_grid(grid, prefix, model_cfg[model_key], **kwargs)

    elif model_key in CONSTANT:
        return build_constant_mf_from_grid(grid, prefix, model_cfg[model_key])

    else:
        raise NotImplementedError(
            f"Invalid multifrequency model: {model_key}"
            f"\n Supported models: {SUPPORTED_MODELS}"
        )
