from astropy import units as u
from astropy.coordinates import Angle

from dataclasses import dataclass
from configparser import ConfigParser


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
            Angle(sky_cfg.get(CENTER_RA_KEY, RA_DEFAULT)),
            Angle(sky_cfg.get(CENTER_DEC_KEY, DEC_DEFAULT))
        )

    @classmethod
    def from_config_parser(cls, sky_cfg: ConfigParser):
        CENTER_RA_KEY = 'image center ra'
        CENTER_DEC_KEY = 'image center dec'

        return SkyCenter(
            Angle(sky_cfg.get(CENTER_RA_KEY, RA_DEFAULT)),
            Angle(sky_cfg.get(CENTER_DEC_KEY, DEC_DEFAULT))
        )
