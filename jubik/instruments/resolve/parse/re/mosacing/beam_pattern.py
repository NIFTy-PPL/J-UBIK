from dataclasses import dataclass


@dataclass
class BeamPatternConfig:
    dish_size: float
    dish_blockage_size: float

    @classmethod
    def from_yaml_dict(cls, yaml: dict):
        SIZE = "size"
        BLOCKAGE = "blockage_size"
        return BeamPatternConfig(
            dish_size=float(yaml[SIZE]), dish_blockage_size=float(yaml[BLOCKAGE])
        )
