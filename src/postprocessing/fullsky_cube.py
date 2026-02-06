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
from astropy.coordinates import SpectralCoord, SkyCoord
from dataclasses import dataclass
from math import e, prod
from os import makedirs
from functools import reduce
from operator import or_

from ..grid import Grid
from ..color import Color, get_spectral_range_index
from ..fits_saver import FitsSaver
from .cube_operations import (
    AX_SAMPLE, AX_STOKES, AX_TIME, AX_SPEC, AX_Y, AX_X, ID_STOKES_I, ID_STOKES_Q, ID_STOKES_U, ID_STOKES_V,
    create_spatial_subgrid_and_extraction_indices, get_stokes_component,
    cube_to_fits,
    cube_spectral_integrated, cube_spatial_integrated, cube_spatial_spectral_integrated,
    cube_spectral_averaged, cube_spatial_averaged, cube_spatial_spectral_averaged,
    cube_stokesI_weighted_velocity_mean, cube_stokesI_weighted_velocity_std, cube_stokesI_weighted_velocity_variance, cube_stokesI_weighted_velocity_skewness, cube_stokesI_weighted_velocity_excess_kurtosis,
    cube_linear_polarization_intensity, cube_fractional_linear_polarization, cube_polarization_angle,
    cube_circular_polarization_fraction, cube_total_polarized_intensity
)

SUPPORTED_MAP_GENERAL = {
    "cube_to_fits": cube_to_fits,
    "spectral_integrated_cube": cube_spectral_integrated, 
    "spatial_integrated_cube": cube_spatial_integrated, 
    "spatial_spectral_integrated_cube": cube_spatial_spectral_integrated,
    "spectral_averaged_cube": cube_spectral_averaged, 
    "spatial_averaged_cube": cube_spatial_averaged,
    "spatial_spectral_averaged_cube": cube_spatial_spectral_averaged,  
}

SUPPORTED_MAP_LINES = {
    "stokesI_weighted_velocity_mean": cube_stokesI_weighted_velocity_mean, 
    "stokesI_weighted_velocity_std": cube_stokesI_weighted_velocity_std,
    "stokesI_weighted_velocity_variance": cube_stokesI_weighted_velocity_variance, 
    "stokesI_weighted_velocity_skewness": cube_stokesI_weighted_velocity_skewness,
    "stokesI_weighted_velocity_excess_kurtosis": cube_stokesI_weighted_velocity_excess_kurtosis,
}

SUPPORTED_MAP_FULL_STOKES = {
    "linear_polarization_intensity": cube_linear_polarization_intensity,
    "fractional_linear_polarization": cube_fractional_linear_polarization,
    "polarization_angle": cube_polarization_angle,
    "circular_polarization_fraction": cube_circular_polarization_fraction,
    "total_polarization_intensity": cube_total_polarized_intensity,
}

SUPPORTED_MAP_MODES_GROUPED = {
    "general": SUPPORTED_MAP_GENERAL,
    "lines": SUPPORTED_MAP_LINES,
    "full_stokes": SUPPORTED_MAP_FULL_STOKES,
}

SUPPORTED_MAP_MODES = reduce(or_, SUPPORTED_MAP_MODES_GROUPED.values())


@dataclass
class FullSkyCube:
    cube_samples: np.ndarray
    grid: Grid
    flux_density_unit: u.Quantity
    reference_frequency: u.Quantity | None = None
    doppler_convention: str = "relativistic"
    prefix: str = ""

    def slice_cube_spatial(self, upper_left_corners, lower_right_corners, frame="icrs"):
        # Creates smaller cubes by slicing rectangular regions defined by the slicing markers
        # The slicing markers are a list of tuple consiting of the coordinates for the upper left and lower right corner
        # Outputs a list of instances of SpatioSpectralCube with spatially smaller cubes
        # corner lists are expected to have the formating "hms dms"
        subcubes = []
        for ulc,lrc in zip(upper_left_corners, lower_right_corners):
            spatial_subgrid, indices_x, indices_y = create_spatial_subgrid_and_extraction_indices(
                spatial_grid = self.grid.spatial,
                upper_left_corner_sky_coord = SkyCoord(ulc, frame=frame),
                lower_right_corner_sky_coord = SkyCoord(lrc, frame=frame)
                )


            subgrid = Grid(
                spatial = spatial_subgrid,
                spectral = self.grid.spectral,
                polarization = self.grid.polarization,
            )

            subcube_samples = np.take(self.cube_samples, indices=indices_y, axis= AX_Y)
            subcube_samples = np.take(subcube_samples, indices=indices_x, axis= AX_X)

            subcubes.append(FullSkyCube(
                cube_samples = subcube_samples,
                grid = subgrid,
                flux_density_unit = self.flux_density_unit,
                reference_frequency = self.reference_frequency,
                doppler_convention = self.doppler_convention,
                prefix = f"{self.prefix}_{ulc}_{lrc}"
            ))
        
        return subcubes

    def slice_cube_spectral(self,spectral_ranges):
        # Spectral pendant to slcie_cube_spectral
        # spectral_ranges should be a list of of two-element list or tuple
        subcubes = []
        for sr in spectral_ranges:
            spec_min = Color(u.Quantity(sr[0]))
            spec_max = Color(u.Quantity(sr[1]))
            spec_min_index = get_spectral_range_index(self.grid.spectral, spec_min)[0]
            spec_max_index = get_spectral_range_index(self.grid.spectral, spec_max)[0]
            indices_spec = np.arange(spec_min_index,spec_max_index + 1)

            subgrid = Grid(
                spatial = self.grid.spatial,
                spectral = self.grid.spectral[spec_min_index : spec_max_index + 1],
                polarization = self.grid.polarization,
            )

            subcube_samples = np.take(self.cube_samples, indices=indices_spec, axis=AX_SPEC)

            subcubes.append(FullSkyCube(
                cube_samples = subcube_samples,
                grid = subgrid,
                flux_density_unit = self.flux_density_unit,
                reference_frequency = self.reference_frequency,
                doppler_convention = self.doppler_convention,
                prefix = f"{self.prefix}_{sr[0]}_{sr[1]}"
            ))

        return subcubes

    def slice_cube_spectral_spatial(self, spectral_ranges, upper_left_corners, lower_right_corners, mode="product", frame="icrs" ):
        # Does slicing in both direction and produces subcubes in to modes:
        # - product: Cartesian product between the spatial and spectral selections
        # - aligned: Associates a spatial and spectral selection. Need to be of same size
        subcubes = []
        if mode == "product":
            spatial_sliced_subcubes = self.slice_cube_spatial(upper_left_corners=upper_left_corners, lower_right_corners=lower_right_corners, frame=frame)

            for cube in spatial_sliced_subcubes:
                subcubes.append(cube.slice_cube_spectral(spectral_ranges))

        elif mode == "aligned":
            for ulc, lrc, sr in zip(upper_left_corners, lower_right_corners,spectral_ranges):
                spatial_subgrid, indices_x, indices_y = create_spatial_subgrid_and_extraction_indices(
                    spatial_grid = self.grid.spatial,
                    upper_left_corner_sky_coord = SkyCoord(ulc, frame=frame),
                    lower_right_corner_sky_coord = SkyCoord(lrc, frame=frame)
                    )

                spec_min = Color(u.Quantity(sr[0]))
                spec_max = Color(u.Quantity(sr[1]))
                spec_min_index = get_spectral_range_index(self.grid.spectral, spec_min)[0]
                spec_max_index = get_spectral_range_index(self.grid.spectral, spec_max)[0]
                indices_spec = np.arange(spec_min_index,spec_max_index + 1)

                subgrid = Grid(
                    spatial = spatial_subgrid,
                    spectral = self.grid.spectral[spec_min_index : spec_max_index + 1],
                    polarization = self.grid.polarization,
                )

                subcube_samples = np.take(self.cube_samples, indices=indices_spec, axis=AX_SPEC)
                subcube_samples = np.take(subcube_samples, indices=indices_y, axis= AX_Y)
                subcube_samples = np.take(subcube_samples, indices=indices_x, axis= AX_X)

                subcubes.append(FullSkyCube(
                    cube_samples = subcube_samples,
                    grid = subgrid,
                    flux_density_unit = self.flux_density_unit,
                    reference_frequency = self.reference_frequency,
                    doppler_convention = self.doppler_convention,
                    prefix = f"{self.prefix}_{ulc}_{lrc}_{sr[0]}_{sr[1]}"
                ))
        
        else:
            raise ValueError("Spatio-spectral slicing only available for the modes 'product' or 'aligned'.")
        
        return subcubes

    @property
    def spatial_pixel_area(self):
        dA = prod(self.grid.spatial.fov)/prod(self.grid.spatial.shape)
        return dA.value, dA.unit

    @property
    def spectral_bin_width(self):
        df = self.grid.spectral.diff()[0]
        return df.value, df.unit

    @property
    def cube_voxel_volume(self):
        dA = prod(self.grid.spatial.fov)/prod(self.grid.spatial.shape)
        df = self.grid.spectral.diff()[0]
        dV = dA*df

        return dV.value, dV.unit

    @property
    def velocities(self):
        if not(isinstance(self.reference_frequency,u.quantity.Quantity)):
            raise ValueError("'reference_frequency' has to be equipped with astropy spectral unit.")

        freqs = SpectralCoord(self.grid.spectral.center.to(self.reference_frequency.unit))
    
        if self.doppler_convention == "relativistic":
            vel = freqs.to(
                unit = u.km/u.s, 
                doppler_rest = self.reference_frequency,
                doppler_convention="relativistic",
                )

            return vel.value, vel.unit
        else:
            raise NotImplementedError("Currently only the most general case is implemented by setting doppler_convention to 'relativistic'.")

    @property
    def stokesI(self):
        return get_stokes_component(self.cube_samples,"I")
    
    @property
    def stokesQ(self):
        return get_stokes_component(self.cube_samples,"Q")

    @property
    def stokesU(self):
        return get_stokes_component(self.cube_samples,"U")

    @property
    def stokesV(self):
        return get_stokes_component(self.cube_samples,"V")

    @classmethod
    def supported_maps(cls):
        for cat, cat_supported_maps in SUPPORTED_MAP_MODES_GROUPED.items():
            print(cat)
            print("-"*50)
            for mode in cat_supported_maps.keys():
                print(mode)
            print(" ")

    def create_maps(self, mode_list: list, output_directory: str, map_units: dict, save_std: bool = True, save_samples: bool = False):
        # Creates all velocity moment maps requested and outputs them as a dictionary.
        # flux_unit hast to be somethink like Jy/(Hz*as**2)
        prefix = "" if self.prefix == "" else f"{self.prefix}_"
        print_prefix = "" if self.prefix == "" else f"{self.prefix} - "

        makedirs(output_directory, exist_ok=True)

        if any(mode not in SUPPORTED_MAP_MODES for mode in mode_list):
            raise ValueError("List contains unsupported mode. Please check with supported modes by calling the 'method supported_maps'.")

        for mode in mode_list:
            if mode in SUPPORTED_MAP_MODES:
                print(f"{print_prefix}{mode}")
                func = SUPPORTED_MAP_MODES[mode]
                field, current_unit, file_name = func(self)

                conversion_factor = current_unit.to(map_units[mode])
                field = field*conversion_factor

                fits = FitsSaver(self.grid,field)
                fits.save_mean(filename=f"{output_directory}/{prefix}{file_name}_mean.fits", sky_unit=map_units[mode])
                if save_std:
                    fits.save_std(filename=f"{output_directory}/{prefix}{file_name}_std.fits", sky_unit=map_units[mode], correct_bias=True)
                if save_samples:
                    fits.save_samples(filename=f"{output_directory}/{prefix}{file_name}_samples.fits", sky_unit=map_units[mode])

    @classmethod
    def build_from_fullskymodel_and_latent_samples(
        cls, 
        full_sky_model: jft.Model, 
        latent_samples_path: str,
        grid: Grid, 
        flux_density_unit: u.Unit,
        reference_frequency: u.Quantity,
        doppler_convention: str,
        prefix: str,
        ):
        import pickle
        with open(latent_samples_path, "rb") as f:
            samples, _ = pickle.load(f)

        sky_samples = np.array(list(full_sky_model(s) for s in samples))

        return cls(
            cube_samples = sky_samples,
            grid = grid,
            flux_density_unit = flux_density_unit,
            reference_frequency = reference_frequency,
            doppler_convention = doppler_convention,
            prefix = prefix,
        )