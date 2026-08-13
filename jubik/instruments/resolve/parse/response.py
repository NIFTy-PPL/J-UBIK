from dataclasses import dataclass
from typing import Optional, Union

import astropy.units as u

from ....parse.parsing_base import StaticTyped

FINUFFT_KEYS = ["finufft"]
DUCC_KEYS = ["ducc", "ducc0"]

EPSILON_KEY = "epsilon"

BACKEND_KEY = "backend"
DO_WGRIDDING_KEY = "do_wgridding"
NTHREADS_KEY = "nthreads"
VERBOSITY_KEY = "verbosity"


@dataclass
class Ducc0Settings(StaticTyped):
    epsilon: float
    do_wgridding: bool
    nthreads: int
    verbosity: int

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict):
        f"""Read ducc0 settings from yaml_dict.

        Parameters
        ----------
        {EPSILON_KEY}: float
        {DO_WGRIDDING_KEY}: bool
        {NTHREADS_KEY}: int
        {VERBOSITY_KEY}: bool
        """
        epsilon = yaml_dict[EPSILON_KEY]
        do_wgridding = yaml_dict[DO_WGRIDDING_KEY]
        nthreads = yaml_dict[NTHREADS_KEY]
        verbosity = yaml_dict[VERBOSITY_KEY]
        return Ducc0Settings(
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=nthreads,
            verbosity=verbosity,
        )


@dataclass
class FinufftSettings(StaticTyped):
    epsilon: float

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict):
        f"""Read finufft settings from yaml_dict.

        Parameters
        ----------
        {EPSILON_KEY}: float
        """
        epsilon = yaml_dict[EPSILON_KEY]
        return FinufftSettings(epsilon=epsilon)


def yaml_to_response_settings(
    response_dict: dict,
) -> Union[Ducc0Settings, FinufftSettings]:
    f"""Read the yaml file in order to parse to Backend settings.
    These can either be `Ducc0Settings` or `FinufftSettings`. 

    Parameters
    ----------
    {BACKEND_KEY}: str (either {FINUFFT_KEYS} or {DUCC_KEYS}.

    Note
    ----
    All other parameters can be seen in `FinufftSettings` or `Ducc0Settings`.
    """

    backend = response_dict[BACKEND_KEY]

    if backend in FINUFFT_KEYS:
        return FinufftSettings.from_yaml_dict(response_dict)

    elif backend in DUCC_KEYS:
        return Ducc0Settings.from_yaml_dict(response_dict)

    raise ValueError(f"Supplied {backend}. Supply either {FINUFFT_KEYS} or {DUCC_KEYS}")
