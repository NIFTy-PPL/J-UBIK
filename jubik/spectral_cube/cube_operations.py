from ..grid import Grid
from ..fits_saver import FitsSaver
from .axes import IntegrationAxis, get_stokes_component, convert_spectral_grid
from .utils import (
    broadcast_to_full,
    intensity_weighted_spectral_moments,
    integrate_cube,
    intensity_weighted_standardized_spectral_moments,
)

import numpy as np
import astropy.units as u
from typing import List
from numpy.typing import NDArray


class CubeOperator:
    def __init__(self, cube_unit, prefix=""):
        self.unit = cube_unit
        self.savefile_prefix = prefix

    def conversion_factor(self, output_unit):
        return self.unit.to(output_unit)

    def __call__(self, cube_samples):
        return cube_samples

    def to_fits(
        self,
        output_directory,
        cube_samples,
        output_unit,
        grid,
        save_std=True,
        save_samples=False,
    ):
        conversion_factor = self.conversion_factor(output_unit=output_unit)
        field = conversion_factor * self.__call__(cube_samples=cube_samples)
        if isinstance(field, u.Quantity):
            field = field.value

        fits = FitsSaver(
            grid=grid,
            field_samples=field,
        )

        fits.save_mean(
            filename=f"{output_directory}/{self.savefile_prefix}_mean.fits",
            sky_unit=output_unit,
        )

        if save_std:
            fits.save_std(
                filename=f"{output_directory}/{self.savefile_prefix}_std.fits",
                sky_unit=output_unit,
                correct_bias=True,
            )

        if save_samples:
            fits.save_samples(
                filename=f"{output_directory}/{self.savefile_prefix}_samples.fits",
                sky_unit=output_unit,
            )


class CubeIntegrate(CubeOperator):
    def __init__(
        self,
        integration_axes: List[IntegrationAxis],
        cube_unit: u.Quantity,
        grid: Grid,
        prefix: str = "",
        mask: NDArray | None = None,
    ):
        if (mask is not None) and (mask.shape != grid.shape):
            raise ValueError("mask must have the same shape as grid.")

        differentials = []
        axs = []
        prefix_adds = []

        for ax in integration_axes:
            differentials.append(ax.differential_elements(grid))
            axs.append(ax.axis)
            prefix_adds.append(ax.label())

        if len(axs) != len(set(axs)):
            raise ValueError("Each integration axis may only appear once.")

        axs_sorted, differentials_sorted, prefix_adds_sorted = zip(
            *sorted(zip(axs, differentials, prefix_adds), key=lambda x: x[0])
        )

        de = None

        for k, sw in enumerate(differentials_sorted):

            if de is not None:
                de = de * broadcast_to_full(sw, k, len(differentials_sorted))
            else:
                de = broadcast_to_full(sw, k, len(differentials_sorted))

        self._bin_widths = de.value if mask is None else de.value * mask[None, :]
        self._axs = tuple(axs_sorted)

        prf = "integrated_" if prefix == "" else f"{prefix}_integrated_"
        super().__init__(
            cube_unit=cube_unit * de.unit, prefix=prf + "_".join(prefix_adds_sorted)
        )

    def __call__(self, cube_samples):
        return integrate_cube(
            cube=cube_samples,
            axes=self._axs,
            deltas=self._bin_widths,
        )


class CubeAverage(CubeOperator):
    def __init__(
        self,
        integration_axes: List[IntegrationAxis],
        cube_unit: u.Quantity,
        grid: Grid,
        prefix: str = "",
        mask: NDArray | None = None,
    ):
        self._integrator = CubeIntegrate(
            integration_axes=integration_axes,
            cube_flux_unit=cube_unit,
            grid=grid,
            prefix=prefix,
            mask=mask,
        )

        bin_widths = self._integrator._bin_widths
        self._vol = np.sum(bin_widths)

        super().__init__(
            cube_unit=cube_unit,
            prefix=self._integrator.prefix.replace("integrated", "averaged"),
        )

    def __call__(self, cube_samples):
        return self._integrator(cube_samples) / self._vol


class SpectralMomentMap(CubeOperator):
    def __init__(
        self, type, frame, grid, doppler_convention=None, reference=None, prefix=""
    ):
        bin_widths, bin_centers = convert_spectral_grid(
            bounds=grid.spectral,
            frame=frame,
            doppler_convention=doppler_convention,
            reference=reference,
        )

        match type:
            case "mean":
                call = lambda cube: intensity_weighted_spectral_moments(
                    1, cube, bin_centers, bin_widths
                )
                unit = bin_widths.unit
            case "standard_deviation":
                call = lambda cube: np.sqrt(
                    intensity_weighted_spectral_moments(
                        2, cube, bin_centers, bin_widths
                    )
                )
                unit = bin_widths.unit
            case "variance":
                call = lambda cube: intensity_weighted_spectral_moments(
                    2, cube, bin_centers, bin_widths
                )
                unit = bin_widths.unit**2
            case "skewness":
                call = lambda cube: intensity_weighted_standardized_spectral_moments(
                    3, cube, bin_centers, bin_widths
                )
                unit = u.dimensionless_unscaled
            case "excess_kurtosis":
                call = (
                    lambda cube: intensity_weighted_standardized_spectral_moments(
                        4, cube, bin_centers, bin_widths
                    )
                    - 3
                )
                unit = u.dimensionless_unscaled
            case _:
                raise NotImplementedError

        self._call = call

        prf = "" if prefix == "" else f"{prefix}_"
        prf += f"intensity_weighted_{frame}_moment_map_{type}"

        if frame == "velocity":
            prf += f"_{str(reference).replace(' ', '')}"

        super().__init__(cube_unit=unit, prefix=prf)

    def __call__(self, cube_samples):
        return self._call(get_stokes_component(cube_samples, "I"))


class LinearPolarization(CubeOperator):
    def __init__(self, cube_unit, prefix=""):
        prf = "" if prefix == "" else f"{prefix}_"
        super().__init__(cube_unit=cube_unit, prefix=prf + "linear_polarization")

    def __call__(self, cube_samples):
        q = get_stokes_component(cube_samples, "Q")
        u = get_stokes_component(cube_samples, "U")
        return np.sqrt(q**2 + u**2)


class FractionalLinearPolarization(CubeOperator):
    def __init__(self, cube_unit, prefix=""):
        prf = "" if prefix == "" else f"{prefix}_"
        super().__init__(
            cube_unit=u.dimensionless_unscaled,
            prefix=prf + "fractional_linear-polarization",
        )

    def __call__(self, cube_samples):
        i = get_stokes_component(cube_samples, "I")
        q = get_stokes_component(cube_samples, "Q")
        u = get_stokes_component(cube_samples, "U")
        return np.sqrt(q**2 + u**2) / i


class PolarizationAngle(CubeOperator):
    def __init__(self, cube_unit, prefix=""):
        prf = "" if prefix == "" else f"{prefix}_"
        super().__init__(cube_unit=u.rad, prefix=prf + "polarization_angle")

    def __call__(self, cube_samples):
        i = get_stokes_component(cube_samples, "I")
        q = get_stokes_component(cube_samples, "Q")
        u = get_stokes_component(cube_samples, "U")
        return 0.5 * np.arctan2(u, q)


class CircularPolarizationFraction(CubeOperator):
    def __init__(self, cube_unit, prefix=""):
        prf = "" if prefix == "" else f"{prefix}_"
        super().__init__(
            cube_unit=u.dimensionless_unscaled,
            prefix=prf + "circular_polarization_fraction",
        )

    def __call__(self, cube_samples):
        i = get_stokes_component(cube_samples, "I")
        v = get_stokes_component(cube_samples, "V")
        return v / i


class TotalPolarizedIntensity(CubeOperator):
    def __init__(self, cube_unit, prefix=""):
        prf = "" if prefix == "" else f"{prefix}_"
        super().__init__(cube_unit=cube_unit, prefix=prf + "total_polarized_intensity")

    def __call__(self, cube_samples):
        q = get_stokes_component(cube_samples, "Q")
        u = get_stokes_component(cube_samples, "U")
        v = get_stokes_component(cube_samples, "V")
        return np.sqrt(q**2 + u**2 + v**2)
