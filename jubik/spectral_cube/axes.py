from ..wcs.wcs_astropy import WcsAstropy
from ..fits_saver import FitsSaver
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame
from math import prod
from enum import IntEnum, Enum
from abc import ABC, abstractmethod
from numpy.typing import NDArray

# Axes and index conventions


class CubeAxes(IntEnum):
    SAMPLES = 0
    STOKES = 1
    TIME = 2
    SPECTRAL = 3
    Y = 4
    X = 5


# Stokes component convention and extraction


class StokesIndices(IntEnum):
    I = 0
    Q = 1
    U = 2
    V = 3


def get_stokes_component(cube_samples, stokes):
    if stokes not in [name for name in StokesIndices.__members__]:
        raise ValueError(
            f"Invalid Stokes component '{stokes}'. Must be either I, Q, U or V."
        )

    idx = StokesIndices[stokes]

    if cube_samples.shape[CubeAxes.STOKES] <= idx:
        raise ValueError(f"Stokes component '{stokes}' not present in this cube.")

    return np.take(cube_samples, [idx], axis=CubeAxes.STOKES)


# Transform spectral grid


def convert_spectral_grid(bounds, frame, doppler_convention=None, reference=None):
    internal_units = {
        "frequency": u.Hz,
        "wavelength": u.m,
        "energy": u.eV,
        "velocity": u.km / u.s,
    }

    if frame not in internal_units.keys():
        raise ValueError(
            f"Invalid spectral frame. Valid arguments are: {internal_units.keys()}"
        )

    if frame == "velocity":
        if doppler_convention is None or reference is None:
            raise ValueError(
                "Velocity conversion requires 'doppler_convention' and 'reference'."
            )

        if not bounds.unit.is_equivalent(reference.unit, equivalencies=u.spectral()):
            raise ValueError(
                "Bounds and reference must be spectrally compatible (Hz, wavelength, energy)."
            )

        doppler_map = {
            "radio": u.doppler_radio(reference),
            "optical": u.doppler_optical(reference),
            "relativistic": u.doppler_relativistic(reference),
        }

        if doppler_convention not in doppler_map:
            raise ValueError(f"Invalid doppler_convention: {doppler_convention}")

        eq = doppler_map[doppler_convention]

    else:
        if not bounds.unit.is_equivalent(u.Hz, equivalencies=u.spectral()):
            raise ValueError(
                "Input bounds must be a spectral coordinate (frequency, wavelength, or energy)."
            )

        eq = u.spectral()

    new_bounds = bounds.to(internal_units[frame], equivalencies=eq)

    widths = np.abs(np.diff(new_bounds, axis=1)[:, 0])
    centers = 0.5 * np.sum(new_bounds, axis=1)

    return widths, centers


# Integration Axes


class IntegrationAxis(ABC):
    @property
    @abstractmethod
    def axis(self) -> int:
        pass

    @abstractmethod
    def differential_elements(self, grid) -> u.Quantity:
        pass

    @abstractmethod
    def label(self) -> str:
        pass


class IntegrationX(IntegrationAxis):
    @property
    def axis(self):
        return CubeAxes.X

    def differential_elements(self, grid):
        return (
            grid.spatial.fov[0] / grid.spatial.shape[0] * np.ones(grid.spatial.shape[0])
        )

    def label(self):
        return "x"


class IntegrationY(IntegrationAxis):
    @property
    def axis(self):
        return CubeAxes.Y

    def differential_elements(self, grid):
        return (
            grid.spatial.fov[1] / grid.spatial.shape[1] * np.ones(grid.spatial.shape[1])
        )

    def label(self):
        return "y"


class IntegrationTime(IntegrationAxis):
    @property
    def axis(self):
        return CubeAxes.TIME

    def differential_elements(self, grid):
        raise NotImplementedError

    def label(self):
        return "t"


class IntegrationSpectral(IntegrationAxis):
    def __init__(self, frame_key, doppler_convention=None, reference=None):
        self._frame = frame_key
        self._doppler = doppler_convention
        self._reference = reference

    @property
    def axis(self):
        return CubeAxes.SPECTRAL

    def differential_elements(self, grid):
        bin_widths, _ = convert_spectral_grid(
            bounds=grid.spectral,
            frame=self._frame,
            doppler_convention=self._doppler,
            reference=self._reference,
        )
        return bin_widths

    def label(self):
        if self._frame == "velocity":
            return f"spec-{self._frame}-{str(self._reference).replace(' ', '')}"
        else:
            return f"spec-{self._frame}"
