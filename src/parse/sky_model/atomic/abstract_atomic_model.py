from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class AtomicModel(ABC):
    @abstractclassmethod
    def from_yaml_dict(cls, yaml_dict: dict):
        ...

    @abstractclassmethod
    def from_config_parser(cls, config: ConfigParser):
        ...
