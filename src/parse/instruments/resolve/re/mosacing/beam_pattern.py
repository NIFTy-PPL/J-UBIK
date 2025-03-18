from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class BeamPatternConfig:
    dish_size: float
    dish_blockage_size: float

    @classmethod
    def from_yaml_dict(
        cls, yaml: dict
    ):
        SIZE = 'size'
        BLOCKAGE = 'blockage_size'
        return BeamPatternConfig(
            dish_size=float(yaml[SIZE]),
            dish_blockage_size=float(yaml[BLOCKAGE]))

    @classmethod
    def from_config_parser(
        cls, beam_pattern_config: ConfigParser,
    ):
        SIZE = 'dish size'
        BLOCKAGE = 'dish blockage size'

        dish_size = float(eval(beam_pattern_config[SIZE]))
        dish_blockage_size = float(eval(beam_pattern_config[BLOCKAGE]))

        return BeamPatternConfig(
            dish_size=dish_size,
            dish_blockage_size=dish_blockage_size)
