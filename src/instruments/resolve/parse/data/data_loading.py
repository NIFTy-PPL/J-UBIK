from dataclasses import dataclass
from os.path import join
from configparser import ConfigParser


def _prepand_path(input_path: str, files: list[str]):
    return [join(input_path, fname) for fname in files]


@dataclass
class DataLoading:
    """Model for data loading.

    Parameters
    ----------
    data_templates: list[str]
        This is the path to the datafiles, with varibable fields for field_ids,
        and spectral_windows.
    field_ids: list[Union[int, None]]
        The field_ids (pointings) for the data_templates, not used if there is
        no corresponding field inside the data_templates.
    spectral_windows: list[Union[int, None]]
        The list of spectral windows for the data_templates. Not used if there
        is no corresponding field inside the data_templates.
    """

    data_templates: list[str]
    field_ids: list[int | None]
    spectral_windows: list[int | None]

    @classmethod
    def from_config_parser(cls, data_cfg: ConfigParser):
        data_templates = _prepand_path(
            data_cfg["data path"], data_cfg["data templates"].split(", ")
        )
        field_ids = data_cfg.get("field ids", None)
        field_ids = eval(field_ids) if field_ids is not None else [None]
        spectral_windows = data_cfg.get("spectral window")
        spectral_windows = (
            [eval(spw) for spw in spectral_windows.split(", ")]
            if spectral_windows is not None
            else [None]
        )

        return DataLoading(
            data_templates=data_templates,
            field_ids=field_ids,
            spectral_windows=spectral_windows,
        )

    @classmethod
    def from_yaml_dict(cls, data_cfg: dict):
        data_templates = _prepand_path(
            data_cfg["data_path"], data_cfg["data_templates"]
        )

        field_ids = data_cfg.get("field_ids", [None])

        spectral = data_cfg.get("spectral")
        spectral_windows = spectral.get("window")

        return DataLoading(
            data_templates=data_templates,
            field_ids=field_ids,
            spectral_windows=spectral_windows,
        )
