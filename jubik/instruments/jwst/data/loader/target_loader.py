from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from nifty.re import logger
from numpy.typing import NDArray


from .....grid import Grid
from ...alignment.star_alignment import StarTables
from ...parse.data.data_loader import IndexAndPath, Subsample
from ...parse.alignment.star_alignment import StarAlignmentConfig
from ...parse.jwst_psf import JwstPsfKernelConfig
from ...parse.masking.data_mask import CornerMasks, NanMaskLoader
from ...alignment.star_alignment import Star
from ...psf.jwst_kernel import load_psf_kernel
from ..jwst_data import DataMetaInformation, JwstData
from ..preloader.data_bounds import DataBounds
from .cutout import DataCutout


# ---------------------
# Loading
# ---------------------


@dataclass(slots=True)
class TargetBundle:
    index: int
    cutout: DataCutout
    subsample_centers: SkyCoord


def load_one_target_bundle(
    index: int,
    jwst_data: JwstData,
    subsample: Subsample | int,
    target_grid: Grid,
    target_data_bounds: DataBounds,
    psf_kernel_configs: JwstPsfKernelConfig,
    corner_masks: CornerMasks | None,
    nan_mask: NanMaskLoader | None,
) -> TargetBundle:
    data, mask, std = jwst_data.bounding_data_mask_std_by_bounding_indices(
        target_data_bounds.bounds[index],
        target_grid.spatial,
        additional_masks_by_corners=corner_masks,
        additional_nan_mask=nan_mask.load_mask()[index]
        if nan_mask is not None
        else None,
    )
    nan_mask = jwst_data.nan_from_bounding_indices(*target_data_bounds.bounds[index])
    psf = load_psf_kernel(
        jwst_data=jwst_data,
        subsample=subsample,
        target_center=target_grid.spatial.center,
        config_parameters=psf_kernel_configs,
    )
    subsample_centers = jwst_data.data_subpixel_centers(
        target_data_bounds.bounds[index],
        subsample=subsample,
    )

    return TargetBundle(
        index=index,
        cutout=DataCutout(data=data, mask=mask, std=std, psf=psf, nan_mask=nan_mask),
        subsample_centers=subsample_centers,
    )


def load_one_target_bundle_from_filepath(
    filepath: IndexAndPath,
    subsample: Subsample | int,
    target_grid: Grid,
    target_data_bounds: DataBounds,
    psf_kernel_configs: JwstPsfKernelConfig,
    corner_masks: CornerMasks,
) -> TargetBundle:
    jwst_data = JwstData(filepath.path)
    return load_one_target_bundle(
        filepath.index,
        jwst_data,
        subsample,
        target_grid,
        target_data_bounds,
        psf_kernel_configs,
        corner_masks,
    )


# ---------------------
# Products
# ---------------------

# TODO : These need to be inhereted from a base-class


@dataclass(slots=True)
class TargetData:
    subsample: int | Subsample
    data: np.ndarray
    mask: np.ndarray
    nan_mask: np.ndarray
    std: np.ndarray
    psf: np.ndarray
    subsample_centers: list[SkyCoord]

    @classmethod
    def from_bundles(
        cls, subsample: int | Subsample, bundles: Iterable[TargetBundle]
    ) -> "TargetData":
        # Ensure deterministic order
        ordered = sorted(bundles, key=lambda tb: tb.index)

        # Collect and stack
        data = np.stack([tb.cutout.data for tb in ordered])
        mask = np.stack([tb.cutout.mask for tb in ordered])
        nan_mask = np.stack([tb.cutout.nan_mask for tb in ordered])
        std = np.stack([tb.cutout.std for tb in ordered])
        psf = np.stack([tb.cutout.psf for tb in ordered])
        centers = [tb.subsample_centers for tb in ordered]

        return cls(
            subsample=subsample,
            data=data,
            mask=mask,
            nan_mask=nan_mask,
            std=std,
            psf=psf,
            subsample_centers=centers,
        )


@dataclass(slots=True)
class TargetDataCore:
    data: np.ndarray
    mask: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.std = np.array(self.std)
        self.mask = np.array(self.mask)
