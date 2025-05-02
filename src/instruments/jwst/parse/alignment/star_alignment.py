from dataclasses import dataclass, field

import astropy.units as u

from ...parse.parametric_model.parametric_prior import (
    ProbabilityConfig,
    prior_config_factory,
)


@dataclass
class FilterAlignmentMeta:
    shape: tuple[int, int]
    fov: u.Quantity
    subsample: int
    star_light_prior: ProbabilityConfig
    library_path: str = ""
    exclude_source_id: list[int] = field(default_factory=list)

    @classmethod
    def from_yaml_dict(cls, raw: dict):
        """Load star alignment from raw.

        Parameters
        ----------
        raw:
            - library_path: str, the path to a library, where to save the found stars
            used for the alignment step.
            - exclude_source_id: list, a list of containing the ids not used in the
            alingment.
            - shape: tuple[int, int], the shape of the star light cutouts.
            - subsample: int, the subsample factor for the star light cutouts.
            - star_light: tuple, the settings for the star light prior.

        """
        shape = raw["shape"]
        subsample = raw["subsample"]
        fov = u.Quantity(raw["fov"])
        starlight = prior_config_factory(raw["star_light"])

        for sh in shape:
            assert sh % 2 != 0, "Need uneven cutouts shape"
        assert subsample % 2 != 0, "Need uneven subsample factor"

        return FilterAlignmentMeta(
            shape=shape,
            fov=fov,
            subsample=subsample,
            star_light_prior=starlight,
            library_path=raw.get("library_path", ""),
            exclude_source_id=raw.get("exclude_source_id", []),
        )
