from dataclasses import dataclass
from itertools import product
from os import curdir
from os.path import join, realpath
from typing import Generator


def _prepand_path(input_path: str, files: list[str]):
    return [join(input_path, fname) for fname in files]


@dataclass(frozen=True)
class DataFile:
    """One loaded data file, as the config refers to it.

    Parameters
    ----------
    path: str
        Canonical path of the file (symlinks resolved before `..`, as the OS
        opens it), the same for every data block that loads it.
    template: str
        Canonical path of the data template the file expands from.
    data_path: str
        The `data_path` of the data block, the root of config entries.
    """

    path: str
    template: str
    data_path: str

    def matches(self, entry: str) -> bool:
        """Whether a config entry names this file or its template.

        Parameters
        ----------
        entry: str
            A file name or data template, relative to `data_path`, or absolute.

        Returns
        -------
        bool
            True if `entry`, joined to `data_path`, is `path` or `template`.
        """
        return realpath(join(self.data_path, entry)) in (self.path, self.template)


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
    data_path: str
        The directory the data_templates are joined to.
    """

    data_templates: list[str]
    field_ids: list[int | None]
    spectral_windows: list[int | None]
    data_path: str = curdir

    @classmethod
    def from_yaml_dict(cls, data_cfg: dict):
        data_templates = _prepand_path(
            data_cfg["data_path"], data_cfg["data_templates"]
        )

        field_ids = data_cfg.get("field_ids", [None])

        spectral = data_cfg.get("spectral", {})
        spectral_windows = spectral.get("window", [None])

        return DataLoading(
            data_templates=data_templates,
            field_ids=field_ids,
            spectral_windows=spectral_windows,
            data_path=data_cfg["data_path"],
        )

    @property
    def data_files(self) -> Generator[DataFile, None, None]:
        """Every data file to load.

        Yields
        ------
        DataFile
            One per combination of field id, data template and spectral window,
            in that nesting order; `{field}` and `{spw}` of the template are
            filled with the field id and the spectral window.
        """
        for fi, dt, spw in product(
            self.field_ids, self.data_templates, self.spectral_windows
        ):
            yield DataFile(
                path=realpath(dt.format(field=fi, spw=spw)),
                template=realpath(dt),
                data_path=self.data_path,
            )
