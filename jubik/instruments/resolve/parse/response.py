from dataclasses import dataclass
from typing import Optional, Union

import astropy.units as u

from ....parse.parsing_base import StaticTyped

FINUFFT_KEYS = ["finufft"]
CUFINUFFT_KEYS = ["cufinufft"]
DUCC_KEYS = ["ducc", "ducc0"]

EPSILON_KEY = "epsilon"

BACKEND_KEY = "backend"
DO_WGRIDDING_KEY = "do_wgridding"
NTHREADS_KEY = "nthreads"
VERBOSITY_KEY = "verbosity"
GPU_MAXBATCHSIZE_KEY = "gpu_maxbatchsize"
UPSAMPFAC_KEY = "upsampfac"


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


@dataclass
class CufinufftSettings(StaticTyped):
    """Radio response through persistent cufinufft plans.

    Plans and point sorts are built lazily during lowering, once per transform
    type, sign and batch size, then reused inside the likelihood. GPU only
    (see ``instruments.resolve.cufinufft``).
    """

    epsilon: float
    gpu_maxbatchsize: int
    upsampfac: float

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict):
        f"""Read cufinufft settings from yaml_dict.

        Parameters
        ----------
        {EPSILON_KEY}: float
        {GPU_MAXBATCHSIZE_KEY}: int, optional
            Grids held per plan; 0 (default) is cufinufft's heuristic
            ``min(n_trans, 8)``, 1 keeps every plan at one grid of memory.
        {UPSAMPFAC_KEY}: float, optional
            Oversampling factor of the fine grid, default 2.0.
        """
        return CufinufftSettings(
            epsilon=float(yaml_dict[EPSILON_KEY]),
            gpu_maxbatchsize=int(yaml_dict.get(GPU_MAXBATCHSIZE_KEY, 0)),
            upsampfac=float(yaml_dict.get(UPSAMPFAC_KEY, 2.0)),
        )


def yaml_to_response_settings(
    response_dict: dict,
) -> Union[Ducc0Settings, FinufftSettings, CufinufftSettings]:
    f"""Read the yaml file in order to parse to Backend settings.
    These can be `Ducc0Settings`, `FinufftSettings` or `CufinufftSettings`.

    Parameters
    ----------
    {BACKEND_KEY}: str ({FINUFFT_KEYS}, {CUFINUFFT_KEYS} or {DUCC_KEYS}).

    Note
    ----
    All other parameters can be seen in `FinufftSettings`, `CufinufftSettings`
    or `Ducc0Settings`.
    """

    backend = response_dict[BACKEND_KEY]

    if backend in FINUFFT_KEYS:
        return FinufftSettings.from_yaml_dict(response_dict)

    elif backend in CUFINUFFT_KEYS:
        return CufinufftSettings.from_yaml_dict(response_dict)

    elif backend in DUCC_KEYS:
        return Ducc0Settings.from_yaml_dict(response_dict)

    raise ValueError(
        f"Supplied {backend}. Supply one of {FINUFFT_KEYS}, {CUFINUFFT_KEYS} or {DUCC_KEYS}"
    )
