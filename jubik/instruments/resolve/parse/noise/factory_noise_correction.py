from .base_line_correction import BaseLineCorrection
from .lower_bound_correction import LowerBoundCorrection

import configparser
from typing import Union


def _return_object_from_config(
    obj: LowerBoundCorrection | BaseLineCorrection,
    data_cfg: configparser.ConfigParser | configparser.SectionProxy | dict,
):
    """Builds object from config parser or yaml_dict."""
    if isinstance(data_cfg, configparser.ConfigParser) or isinstance(
        data_cfg, configparser.SectionProxy
    ):
        return obj.from_config_parser(data_cfg)

    elif isinstance(data_cfg, dict):
        return obj.from_yaml_dict(data_cfg)


def factory_noise_correction_parser(
    data_cfg: configparser.ConfigParser | configparser.SectionProxy | dict,
) -> Union[LowerBoundCorrection, BaseLineCorrection] | None:

    if isinstance(data_cfg, configparser.ConfigParser) or isinstance(
        data_cfg, configparser.SectionProxy
    ):
        if "noise correction model" not in data_cfg:
            return None
        model_name = data_cfg["noise correction model"].lower()

    elif isinstance(data_cfg, dict):
        raise NotImplementedError

    match model_name:
        case "lowerbound":
            return _return_object_from_config(LowerBoundCorrection, data_cfg)
        case "antennabased":
            return _return_object_from_config(BaseLineCorrection, data_cfg)

    raise ValueError('Supported models: ["lowerbound", "antennabased"]')
