from .abstract_hybrid_model import HybridModel

from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class AdditiveModel(HybridModel):

    def from_yaml_dict(cls, yaml_dict: dict):

    def from_config_parser(cls, config: ConfigParser):
        ...
