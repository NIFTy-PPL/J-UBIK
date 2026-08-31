from .base_line_correction import BaseLineCorrection


def factory_noise_correction_parser(
    data_cfg: dict,
) -> BaseLineCorrection | None:
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
        case "baseline":
            return BaseLineCorrection.from_yaml_dict(model[model_name])

    raise ValueError(f'Unknown model "{model_name}". Supported models: ["baseline"]')
