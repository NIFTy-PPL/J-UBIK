from .base_line_correction import BaseLineCorrection
from .lower_bound_correction import LowerBoundCorrection

import configparser
from typing import Union


def factory_noise_correction_parser(
    data_cfg: dict,
) -> Union[LowerBoundCorrection, BaseLineCorrection] | None:
    """Parse the noise correction model.

    Parameters
    ----------
    data_cfg: dict
        - `weight_correction` w
    """

    model = data_cfg.get("weight_correction", None)

    if model is None:
        return None

    match model_name := next(iter(model.keys())):
        case "lowerbound":
            return LowerBoundCorrection.from_yaml_dict(model[model_name])
        case "baseline":
            return BaseLineCorrection.from_yaml_dict(model[model_name])

    raise ValueError('Supported models: ["lowerbound", "baseline"]')
