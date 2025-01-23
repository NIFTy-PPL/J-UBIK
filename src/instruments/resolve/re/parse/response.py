from jubik0.instruments.jwst.grid import Grid

import astropy.units as u

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SkyDomain:
    # Polarization
    polarization_labels: list[str]
    # Time
    times: list[float]
    # Frequencies
    frequencies: list[float]
    # Spatial
    npix_x: int
    npix_y: int
    pixsize_x: float
    pixsize_y: float
    # Optional
    center_x: Optional[float] = 0.
    center_y: Optional[float] = 0.


def sky_domain_from_grid(
    grid: Grid,
    center: Optional[tuple[float]] = [0., 0.]
) -> SkyDomain:
    SPECTRAL_UNIT = u.Hz
    SPATIAL_UNIT = u.rad

    shape = grid.spatial.shape
    distances = grid.spatial.distances_in(SPATIAL_UNIT)
    frequencies = grid.spectral.binbounds_in(SPECTRAL_UNIT)

    return SkyDomain(
        polarization_labels=grid.polarization_labels,
        times=grid.times,
        frequencies=frequencies,
        npix_x=shape[0],
        npix_y=shape[1],
        pixsize_x=distances[0],
        pixsize_y=distances[1],
        center_x=center[0],
        center_y=center[1],
    )


BACKEND_KEY = 'backend'
FINUFFT_KEYS = ['finufft']
DUCC_KEYS = ['ducc', 'ducc0']
EPSILON_KEY = 'epsilon'
POLARIZATION_KEY = 'no_polarization'

DO_WGRIDDING_KEY = 'do_wgridding'
NTHREADS_KEY = 'nthreads'
VERBOSITY_KEY = 'verbosity'


@dataclass
class Ducc0Settings:
    epsilon: float
    do_wgridding: bool
    nthreads: int
    verbosity: int


@dataclass
class FinufftSettings:
    epsilon: float
    no_polarization: bool=False


def yaml_to_response_settings(
    response_dict: dict
) -> Union[Ducc0Settings, FinufftSettings]:
    '''Read the yaml file in order to parse to Backend settings.
    These can either be `Ducc0Settings` or `FinufftSettings`. '''

    backend = response_dict[BACKEND_KEY]

    if backend in FINUFFT_KEYS:
        epsilon = response_dict[EPSILON_KEY]
        no_polarization = response_dict[POLARIZATION_KEY]
        return FinufftSettings(
            epsilon=epsilon,
            no_polarization=no_polarization
        )

    elif backend in DUCC_KEYS:
        epsilon = response_dict[EPSILON_KEY]
        do_wgridding = response_dict[DO_WGRIDDING_KEY]
        nthreads = response_dict[NTHREADS_KEY]
        verbosity = response_dict[VERBOSITY_KEY]
        return Ducc0Settings(
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=nthreads,
            verbosity=verbosity,
        )
