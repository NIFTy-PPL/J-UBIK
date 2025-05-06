from dataclasses import dataclass, field
from typing import Iterator, Union

import numpy as np
from astropy.coordinates import SkyCoord

from ..data.jwst_data import DataMetaInformation


@dataclass
class TargetData:
    meta: DataMetaInformation | None = None
    data: np.ndarray | list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | list[np.ndarray] = field(default_factory=list)
    std: np.ndarray | list[np.ndarray] = field(default_factory=list)
    psf: np.ndarray | list[np.ndarray] = field(default_factory=list)
    subsample_centers: SkyCoord | list[SkyCoord] = field(default_factory=list)
    star_in_subsampled_pixels: list[np.ndarray | None] = field(default_factory=list)
    observation_ids: list[int | None] = field(default_factory=list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a *new* TargetData with the chosen index/indices.
        Works for both integer indices and slices.
        """
        if isinstance(idx, slice):
            rng = range(*idx.indices(len(self)))
            return TargetData(
                data=[self.data[i] for i in rng],
                mask=[self.mask[i] for i in rng],
                std=[self.std[i] for i in rng],
                psf=[self.psf[i] for i in rng],
                meta=[self.meta[i] for i in rng],
                subsample_centers=[self.subsample_centers[i] for i in rng],
                correction_prior=[self.correction_prior[i] for i in rng],
                star_in_subsampled_pixels=[
                    self.star_in_subsampled_pixels[i] for i in rng
                ],
                observation_ids=[self.observation_ids[i] for i in rng],
            )

        return TargetData(
            data=self.data[idx],
            mask=self.mask[idx],
            std=self.std[idx],
            psf=self.psf[idx],
            meta=self.meta[idx],
            subsample_centers=self.subsample_centers[idx],
            correction_prior=self.correction_prior[idx],
            star_in_subsampled_pixels=self.star_in_subsampled_pixels[idx],
            observation_ids=self.observation_ids[idx],
        )

    def __iter__(self) -> Iterator["TargetData"]:
        """
        Iterate over the TargetData row-by-row, yielding a *new*
        TargetData instance at each step that contains exactly one
        element per field.
        """
        for i in range(len(self)):
            yield self[i]

    def add_or_check_meta_data(self, filter_data_meta: DataMetaInformation):
        if self.meta is None:
            self.meta = filter_data_meta
        else:
            assert self.meta == filter_data_meta

    def append_observation(
        self,
        meta: DataMetaInformation,
        data: np.ndarray,
        mask: np.ndarray,
        std: np.ndarray,
        subsample_centers: SkyCoord,
        psf: np.ndarray,
        star_in_subsampled_pixles: np.ndarray | None = None,
        observation_id: int | None = None,
    ):
        self.add_or_check_meta_data(meta)
        self.data.append(data)
        self.mask.append(mask)
        self.std.append(std)
        self.subsample_centers.append(subsample_centers)
        self.psf.append(psf)
        self.star_in_subsampled_pixels.append(star_in_subsampled_pixles)
        self.observation_ids.append(observation_id)


@dataclass
class StarData:
    meta: DataMetaInformation | None = None
    data: np.ndarray | list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | list[np.ndarray] = field(default_factory=list)
    std: np.ndarray | list[np.ndarray] = field(default_factory=list)
    psf: np.ndarray | list[np.ndarray] = field(default_factory=list)
    sky_array: list[np.ndarray] = field(default_factory=list)
    star_in_subsampled_pixels: list[np.ndarray | None] = field(default_factory=list)
    observation_ids: list[int | None] = field(default_factory=list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a *new* StarData with the chosen index/indices.
        Works for both integer indices and slices.
        """
        if isinstance(idx, slice):
            rng = range(*idx.indices(len(self)))
            return StarData(
                data=[self.data[i] for i in rng],
                mask=[self.mask[i] for i in rng],
                std=[self.std[i] for i in rng],
                psf=[self.psf[i] for i in rng],
                meta=[self.meta[i] for i in rng],
                sky_array=[self.sky_array[i] for i in rng],
                correction_prior=[self.correction_prior[i] for i in rng],
                star_in_subsampled_pixels=[
                    self.star_in_subsampled_pixels[i] for i in rng
                ],
                observation_ids=[self.observation_ids[i] for i in rng],
            )

        return StarData(
            data=self.data[idx],
            mask=self.mask[idx],
            std=self.std[idx],
            psf=self.psf[idx],
            meta=self.meta[idx],
            sky_array=self.sky_array[idx],
            correction_prior=self.correction_prior[idx],
            star_in_subsampled_pixels=self.star_in_subsampled_pixels[idx],
            observation_ids=self.observation_ids[idx],
        )

    def __iter__(self) -> Iterator["StarData"]:
        """
        Iterate over the StarData row-by-row, yielding a *new*
        StarData instance at each step that contains exactly one
        element per field.
        """
        for i in range(len(self)):
            yield self[i]

    def add_or_check_meta_data(self, filter_data_meta: DataMetaInformation):
        if self.meta is None:
            self.meta = filter_data_meta
        else:
            assert self.meta == filter_data_meta

    def append_observation(
        self,
        meta: DataMetaInformation,
        data: np.ndarray,
        mask: np.ndarray,
        std: np.ndarray,
        sky_array: SkyCoord,
        psf: np.ndarray,
        star_in_subsampled_pixles: np.ndarray | None = None,
        observation_id: int | None = None,
    ):
        self.add_or_check_meta_data(meta)
        self.data.append(data)
        self.mask.append(mask)
        self.std.append(std)
        self.sky_array.append(sky_array)
        self.psf.append(psf)
        self.star_in_subsampled_pixels.append(star_in_subsampled_pixles)
        self.observation_ids.append(observation_id)
