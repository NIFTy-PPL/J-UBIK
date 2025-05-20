from dataclasses import dataclass, field

import astropy.units as u

from ...parse.parametric_model.parametric_prior import (
    ProbabilityConfig,
    prior_config_factory,
)


@dataclass
class StarAlignmentConfig:
    """Parser for the StarAlignment.

    Parameters
    ----------
    fov: u.Quantity
        Field of view for the star cutouts.
    subsample: int
        Subsample factor for the star cutouts.
    star_light_prior: ProbabilityConfig
        The prior probability for the Star light.
    library_path: str
        where to save the star tables
    exclude_source_id: list[int]
        which sources to exclude from the table query.
    """

    fov: u.Quantity
    subsample: int
    star_light_prior: ProbabilityConfig
    library_path: str = ""
    exclude_source_id: list[int] = field(default_factory=list)

    @classmethod
    def from_yaml_dict(cls, raw: dict | None):
        """Load star alignment from raw.

        Parameters
        ----------
        raw:
            - fov:
            - subsample: int, the subsample factor for the star light cutouts.
            - star_light: tuple, the settings for the star light prior.
            - library_path: str, the path to a library, where to save the found stars
            used for the alignment step.
            - exclude_source_id: list, a list of containing the ids not used in the
            alingment.
        """
        if raw is None:
            return None

        return StarAlignmentConfig(
            fov=u.Quantity(raw["fov"]),
            subsample=raw["subsample"],
            star_light_prior=prior_config_factory(raw["star_light"]),
            library_path=raw.get("library_path", ""),
            exclude_source_id=raw.get("exclude_source_id", []),
        )
