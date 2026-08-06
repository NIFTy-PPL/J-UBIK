# What should be included into the mapping operations:
# - Spatially integrated -> moment 0 map in casa
# - Velcoity map -> moment 1 map in casa
# - Dispersion map -> moment 2 map in casa
# - masking or slicing operation so that only a subset can be created
# - fits saving as well as npz saving
# - to obey right statistics one should do everything sample input and intensity model/line input is needed
# - skewness map which is statistically a moment 3 map

# Shape of cubes should follow the convention (spectral, spatial, spatial)

import nifty.re as jft
import numpy as np
import astropy.units as u

from astropy.coordinates import SkyCoord
from dataclasses import dataclass
from typing import List
from os import makedirs

# from .slice_cube import slice_cube_spatial, slice_cube_spectral
from .axes import IntegrationX, IntegrationY, IntegrationSpectral, IntegrationTime
from ..grid import Grid

from .cube_operations import (
    CubeOperator,
    CubeIntegrate,
    CubeAverage,
    SpectralMomentMap,
    LinearPolarization,
    FractionalLinearPolarization,
    PolarizationAngle,
    CircularPolarizationFraction,
    TotalPolarizedIntensity,
)


def setup_integrator_averager(
    averaging,
    integration_axes,
    cube_unit,
    grid,
    doppler_convention=None,
    reference=None,
    prefix="",
):
    axs = []
    for iax in integration_axes:
        match iax["name"]:
            case "spatial_x":
                ax = IntegrationX()
            case "spatial_y":
                ax = IntegrationY()
            case "spectral":
                ax = IntegrationSpectral(
                    frame_key=iax["frame"],
                    doppler_convention=doppler_convention,
                    reference=reference,
                )
            case "temporal":
                ax = IntegrationTime()
        axs.append(ax)

    if averaging:
        return CubeAverage(
            integration_axes=axs,
            cube_unit=cube_unit,
            grid=grid,
            prefix=prefix,
        )
    else:
        return CubeIntegrate(
            integration_axes=axs,
            cube_unit=cube_unit,
            grid=grid,
            prefix=prefix,
        )


@dataclass
class FullSkyCube:
    cube_samples: np.ndarray
    grid: Grid
    flux_density_unit: u.Quantity
    reference: u.Quantity | None = None
    doppler_convention: str | None = None
    prefix: str = ""

    def create_maps(self, map_configs: List[dict], output_directory: str):
        makedirs(output_directory, exist_ok=True)
        for cfg in map_configs:

            match cfg["operation"]:
                case "cube":
                    op = CubeOperator(
                        cube_unit=self.flux_density_unit, prefix=self.prefix
                    )
                case "integrate":
                    op = setup_integrator_averager(
                        averaging=False,
                        integration_axes=cfg["axes"],
                        cube_unit=self.flux_density_unit,
                        grid=self.grid,
                        doppler_convention=self.doppler_convention,
                        reference=self.reference,
                        prefix=self.prefix,
                    )
                case "average":
                    op = setup_integrator_averager(
                        averaging=True,
                        integration_axes=cfg["axes"],
                        cube_unit=self.flux_density_unit,
                        grid=self.grid,
                        doppler_convention=self.doppler_convention,
                        reference=self.reference,
                        prefix=self.prefix,
                    )
                case "spectral_moment":
                    op = SpectralMomentMap(
                        type=cfg["type"],
                        frame=cfg["frame"],
                        grid=self.grid,
                        doppler_convention=self.doppler_convention,
                        reference=self.reference,
                        prefix=self.prefix,
                    )
                case _:
                    raise NotImplementedError

            op.to_fits(
                output_directory=output_directory,
                cube_samples=self.cube_samples,
                output_unit=cfg["output_unit"],
                grid=self.grid,
                save_std=cfg.get("save_std", True),
                save_samples=cfg.get("save_samples", False),
            )

    @classmethod
    def build_from_fullskymodel_and_latent_samples(
        cls,
        full_sky_model: jft.Model,
        latent_samples_path: str,
        grid: Grid,
        flux_density_unit: u.Unit,
        reference: u.Quantity,
        doppler_convention: str,
        prefix: str,
    ):
        import pickle

        with open(latent_samples_path, "rb") as f:
            samples, _ = pickle.load(f)

        sky_samples = np.array(list(full_sky_model(s) for s in samples))

        return cls(
            cube_samples=sky_samples,
            grid=grid,
            flux_density_unit=flux_density_unit,
            reference=reference,
            doppler_convention=doppler_convention,
            prefix=prefix,
        )
