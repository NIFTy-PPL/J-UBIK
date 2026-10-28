from dataclasses import dataclass, field

import astropy.units as u

from .....parse.models.distributions import (
    ProbabilityConfig,
    prior_config_factory,
)
from ...data.jwst_information import JWST_FILTERS

DEFAULT_KEY = "default"


def _fov_angle(key: str, value) -> u.Quantity:
    try:
        return u.Quantity(value).to(u.arcsec)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"gaia_alignment.fov{key}: needs an angle like `0.7arcsec`, got {value!r}"
        ) from error


@dataclass
class StarAlignmentConfig:
    """Parser for the StarAlignment.

    Parameters
    ----------
    fov: u.Quantity | dict[str, u.Quantity]
        Field of view for the star cutouts: one quantity for every filter, or
        a mapping with a `default` and filter-name entries, see `fov_of`.
    subsample: int
        Subsample factor for the star cutouts.
    star_light_prior: ProbabilityConfig
        The prior probability for the Star light.
    library_path: str
        where to save the star tables
    exclude_source_id: list[int]
        which sources to exclude from the table query.
    position_error: bool
        Give every star a position nuisance with the Gaia error as prior.
    """

    fov: u.Quantity | dict[str, u.Quantity]
    subsample: int
    star_light_prior: ProbabilityConfig
    library_path: str = ""
    exclude_source_id: list[int] = field(default_factory=list)
    position_error: bool = False

    def fov_of(self, filter_name: str) -> u.Quantity:
        """The cutout size for `filter_name`, any case: the single quantity, or the filter entry, else `default`."""
        if isinstance(self.fov, dict):
            return self.fov.get(filter_name.lower(), self.fov[DEFAULT_KEY])
        return self.fov

    @classmethod
    def from_yaml_dict(cls, raw: dict | None):
        """Load star alignment from raw.

        Parameters
        ----------
        raw:
            - fov: a quantity string (`0.7arcsec`), or a mapping of filter name
              to quantity with a `default` entry for the filters not named.
            - subsample: int, the subsample factor for the star light cutouts.
            - star_light: tuple, the settings for the star light prior.
            - library_path: str, the path to a library, where to save the found stars
            used for the alignment step.
            - exclude_source_id: list, a list of containing the ids not used in the
            alingment.
            - position_error: bool, give every star a latent offset from its Gaia
            position, prior normal(0, Gaia error at the exposure epoch). Default False.
        """
        if raw is None:
            return None

        fov = raw["fov"]
        if isinstance(fov, dict):
            if DEFAULT_KEY not in fov:
                raise ValueError(
                    f"gaia_alignment.fov: a mapping needs a `{DEFAULT_KEY}` entry "
                    f"for the filters it does not name, got {sorted(fov)}"
                )
            fov = {
                name.lower(): _fov_angle(f".{name}", value) for name, value in fov.items()
            }
            unknown = sorted(set(fov) - {DEFAULT_KEY} - {f.lower() for f in JWST_FILTERS})
            if unknown:
                raise ValueError(
                    f"gaia_alignment.fov: unknown filter names {unknown}, "
                    "a typo would silently fall back to `default`"
                )
        else:
            fov = _fov_angle("", fov)

        return StarAlignmentConfig(
            fov=fov,
            subsample=raw["subsample"],
            star_light_prior=prior_config_factory(raw["star_light"]),
            library_path=raw.get("library_path", ""),
            exclude_source_id=raw.get("exclude_source_id", []),
            position_error=raw.get("position_error", False),
        )
