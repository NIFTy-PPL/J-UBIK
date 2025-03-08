from abc import ABC, abstractclassmethod
from dataclasses.abc import abstractproperty

from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class HybridModel(ABC):
    @abstractproperty
    children: list

    @abstractclassmethod
    def from_yaml_dict(cls, yaml_dict: dict):
        ...

    @abstractclassmethod
    def from_config_parser(cls, config: ConfigParser):
        ...


@dataclass
class A(HybridModel):
    children: list[str]

    @classmethod
    def from_yaml_dict(cls, a):
        return A(1)
