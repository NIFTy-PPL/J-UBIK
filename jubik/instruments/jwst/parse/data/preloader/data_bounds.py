from collections import UserDict
import numpy as np


class DataBoundsAdjust(UserDict):
    @classmethod
    def from_yaml_dict(cls, raw: dict | None) -> "DataBoundsAdjust":
        if raw is None:
            return cls({})

        return cls({key: np.array(val) for key, val in raw.items()})
