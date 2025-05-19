from dataclasses import dataclass
from ..jwst_data import DataMetaInformation


@dataclass
class FilterConsistency:
    meta: DataMetaInformation | None = None

    def check_meta_consistency(self, meta: DataMetaInformation, filepath: str) -> None:
        if self.meta is None:
            self.meta = meta

        assert self.meta == meta, f"{filepath} is not consistent with previous file."
