from .angle import parse_angle

from astropy import units as u

from dataclasses import dataclass


RA_DEFAULT = 0.*u.hourangle
DEC_DEFAULT = 0.*u.deg


@dataclass
class SkyCenter:
    ra: u.Quantity
    dec: u.Quantity

    @classmethod
    def from_yaml_dict(
        cls,
        sky_cfg: dict,
    ):
        CENTER_RA_KEY = 'ra'
        CENTER_DEC_KEY = 'dec'

        return SkyCenter(
            parse_angle(sky_cfg, CENTER_RA_KEY, RA_DEFAULT),
            parse_angle(sky_cfg, CENTER_DEC_KEY, DEC_DEFAULT)
        )
