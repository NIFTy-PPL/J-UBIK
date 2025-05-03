from ...likelihood import build_gaussian_likelihood
from ...grid import Grid
from ...wcs.wcs_jwst_data import WcsJwstData
from .jwst_response import build_jwst_response
from .data.jwst_data import JwstData, DataMetaInformation
from .data.data_loading import DataBoundsPreloading
from .jwst_psf import load_psf_kernel
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .plotting.residuals import ResidualPlottingInformation
from .config_handler import (
    get_grid_extension_from_config,
)
from ..gaia.star_finder import load_gaia_stars_in_fov, join_tables
from .plotting.plotting_sky import plot_sky_coords, plot_jwst_panels
from .alignment.star_alignment import FilterAlignment
from .alignment.star_fields import build_stars

from .zero_flux_model import build_zero_flux_model

# Parsing
from .parse.jwst_psf import yaml_to_psf_kernel_config
from .parse.zero_flux_model import yaml_to_zero_flux_prior_config
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.jwst_response import SkyMetaInformation
from .parse.masking.data_mask import yaml_to_corner_mask_configs
from .parse.alignment.star_alignment import FilterAlignmentMeta
from .parse.data.data_loading import DataFilePaths


# Libraries
import jax.numpy as jnp
import nifty8.re as jft
from nifty8.logger import logger
import numpy as np

# std
from functools import reduce
from astropy import units as u
from astropy.coordinates import SkyCoord
from typing import Union, Iterator
from dataclasses import dataclass, field


def plot_test(filter_alignment, ii, jwst_data):
    import astropy
    from functools import partial
    import matplotlib.pyplot as plt

    stars = filter_alignment.get_stars(ii)
    plot_position_stars = partial(
        plot_sky_coords,
        sky_coords=[star.position for star in stars],
        labels=[star.id for star in stars],
        behavior_index=lambda index, sky_coords: (
            sky_coords if index == 0 else [sky_coords[index - 1]]
        ),
    )
    jwst_wcs = astropy.wcs.WCS(jwst_data.wcs.to_header())

    shape = (
        int(
            (
                filter_alignment.alignment_meta.fov
                / jwst_data.meta.pixel_distance.to(
                    filter_alignment.alignment_meta.fov.unit
                )
            ).value
        ),
    ) * 2

    data = [jwst_data.dm.data]
    wcs = [jwst_wcs]
    for star in stars:
        minx, maxx, miny, maxy = star.bounding_indices(jwst_data, shape)
        print(star.id, minx, maxx, miny, maxy)
        wcs.append(jwst_wcs[miny : maxy + 1, minx : maxx + 1])
        data.append(jwst_data.data_inside_extrema((minx, maxx, miny, maxy)))

    mean = np.nanmean(jwst_data.dm.data)
    fig, axs = plot_jwst_panels(
        data,
        wcs,
        nrows=1,
        ncols=len(data),
        vmin=0.9 * mean,
        vmax=1.1 * mean,
        coords_plotter=plot_position_stars,
    )


@dataclass
class FilterData:
    meta: DataMetaInformation | None = None
    data: np.ndarray | list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | list[np.ndarray] = field(default_factory=list)
    std: np.ndarray | list[np.ndarray] = field(default_factory=list)
    psf: np.ndarray | list[np.ndarray] = field(default_factory=list)
    subsample_centers: SkyCoord | list[SkyCoord] = field(default_factory=list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a *new* FilterData with the chosen index/indices.
        Works for both integer indices and slices.
        """
        if isinstance(idx, slice):
            rng = range(*idx.indices(len(self)))
            return FilterData(
                data=[self.data[i] for i in rng],
                mask=[self.mask[i] for i in rng],
                std=[self.std[i] for i in rng],
                psf=[self.psf[i] for i in rng],
                meta=[self.meta[i] for i in rng],
                subsample_centers=[self.subsample_centers[i] for i in rng],
                correction_prior=[self.correction_prior[i] for i in rng],
            )

        return FilterData(
            data=self.data[idx],
            mask=self.mask[idx],
            std=self.std[idx],
            psf=self.psf[idx],
            meta=self.meta[idx],
            subsample_centers=self.subsample_centers[idx],
            correction_prior=self.correction_prior[idx],
        )

    def __iter__(self) -> Iterator["FilterData"]:
        """
        Iterate over the FilterData row-by-row, yielding a *new*
        FilterData instance at each step that contains exactly one
        element per field.
        """
        for i in range(len(self)):
            yield self[i]

    def add_and_check_meta_data(self, filter_data_meta: DataMetaInformation):
        if self.meta is None:
            self.meta = filter_data_meta
        else:
            assert self.meta == filter_data_meta


def build_jwst_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_model: jft.Model,
    sky_key: str = "sky",
    files_key: str = "files",
    telescope_key: str = "telescope",
    sky_unit: u.Unit | None = None,
) -> Union[jft.Likelihood, FilterProjector, dict]:
    """Build the jwst likelihood according to the config and grid."""

    filter_projector = build_filter_projector(
        sky_model, grid, cfg[files_key]["filter"].keys()
    )

    # Parsing
    zero_flux_prior_configs = yaml_to_zero_flux_prior_config(
        cfg[telescope_key]["zero_flux"]
    )
    psf_kernel_configs = yaml_to_psf_kernel_config(cfg[telescope_key]["psf"])
    rotation_and_shift_algorithm = rotation_and_shift_algorithm_config_factory(
        cfg[telescope_key]["rotation_and_shift"]
    )
    data_subsample = cfg[telescope_key]["rotation_and_shift"]["subsample"]
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
    )

    gaia_alignment_meta_data = FilterAlignmentMeta.from_yaml_dict(
        cfg[telescope_key].get("gaia_alignment", {})
    )

    data_paths = DataFilePaths.from_yaml_dict(cfg[files_key])

    target_plotting = ResidualPlottingInformation()
    likelihoods = []
    for filter_and_files in data_paths.filters:
        filter_data = FilterData()

        target_preloading = DataBoundsPreloading()

        filter_alignment = FilterAlignment(
            filter_name=filter_and_files.name, alignment_meta=gaia_alignment_meta_data
        )
        filter_alignment.load_correction_prior(
            cfg[telescope_key]["rotation_and_shift"]["correction_priors"],
            number_of_observations=len(filter_and_files.filepaths),
        )

        for ii, filepath in enumerate(filter_and_files.filepaths):
            print(ii, filepath)
            jwst_data = JwstData(filepath, subsample=data_subsample)

            # filter_alignment.star_tables.append(
            #     load_gaia_stars_in_fov(
            #         jwst_data.wcs.world_corners(),
            #         filter_alignment.alignment_meta.library_path,
            #     )
            # )

            target_preloading.append_shapes_and_bounds(
                jwst_data,
                sky_corners=grid.spatial.world_corners(
                    extension_value=sky_meta.grid_extension
                ),
            )

            if False:
                plot_test(filter_alignment, ii, jwst_data)

        target_preloading = target_preloading.align_shapes()

        # stars = build_stars(
        #     filter_and_files.name,
        #     shape=filter_alignment.alignment_meta.shape,
        #     star_ids=[star.id for star in filter_alignment.get_stars()],
        #     star_light_prior=filter_alignment.alignment_meta.star_light_prior,
        # )
        # from jax import random
        #
        # key = random.PRNGKey(42)
        # stars(stars.init(key))
        # list(stars._target_ids)

        for ii, filepath in enumerate(filter_and_files.filepaths):
            logger.info(f"Loading: {filter_and_files.name} {ii} {filepath}")

            # Loading data, std, and mask.
            jwst_data = JwstData(filepath, subsample=data_subsample)
            energy_name = filter_projector.get_key(jwst_data.pivot_wavelength)
            if ii == 0:
                previous_energy_name = energy_name
            else:
                assert energy_name == previous_energy_name

            data, mask, std, data_subsampled_centers = (
                jwst_data.bounding_data_mask_std_subpixel_by_bounding_indices(
                    grid.spatial,
                    target_preloading.bounding_indices[ii],
                    yaml_to_corner_mask_configs(cfg[telescope_key]),
                )
            )
            print(data.shape, data_subsampled_centers.shape)

            psf = load_psf_kernel(
                jwst_data=jwst_data,
                target_center=grid.spatial.center,
                config_parameters=psf_kernel_configs,
            )

            filter_data.add_and_check_meta_data(jwst_data.meta)
            filter_data.data.append(data)
            filter_data.mask.append(mask)
            filter_data.std.append(std)
            filter_data.subsample_centers.append(data_subsampled_centers)
            filter_data.psf.append(psf)
            filter_alignment.boresight.append(jwst_data.get_boresight_world_coords())

        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter_and_files.name,
            correction_prior=filter_alignment.correction_prior,
            rotation_center=SkyCoord(filter_alignment.boresight),
        )

        zero_flux_model = build_zero_flux_model(
            filter_and_files.name,
            zero_flux_prior_configs.get_name_setting_or_default(filter_and_files.name),
            shape=(len(filter_data), 1, 1),
        )

        jwst_target_response = build_jwst_response(
            sky_domain={energy_name: filter_projector.target[energy_name]},
            data_subsampled_centers=filter_data.subsample_centers,
            data_meta=filter_data.meta,
            sky_wcs=grid.spatial,
            sky_meta=sky_meta,
            rotation_and_shift_algorithm=rotation_and_shift_algorithm,
            shift_and_rotation_correction=shift_and_rotation_correction,
            psf=np.array(filter_data.psf),
            zero_flux_model=zero_flux_model,
            data_mask=np.array(filter_data.mask),
        )

        likelihood = build_gaussian_likelihood(
            np.array(filter_data.data)[np.array(filter_data.mask)],
            np.array(filter_data.std)[np.array(filter_data.mask)],
        )
        likelihood = likelihood.amend(
            jwst_target_response, domain=jft.Vector(jwst_target_response.domain)
        )

        target_plotting.append_information(
            filter=filter_and_files.name,
            data=np.array(filter_data.data),
            std=np.array(filter_data.std),
            mask=np.array(filter_data.mask),
            model=jwst_target_response,
        )
        likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x + y, likelihoods)

    return likelihood, filter_projector, target_plotting
