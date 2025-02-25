from astropy import units as u

from dataclasses import dataclass
from configparser import ConfigParser


RA_DEFAULT = 0.*u.hourangle
DEC_DEFAULT = 0.*u.deg


@dataclass
class SkyCenter:
    ra: u.Quantity
    dec: u.Quantity

    @staticmethod
    def _get_quantity(
        sky_cfg: dict | ConfigParser,
        key: str,
        default: u.Quantity
    ) -> u.Quantity:
        val = u.Quantity(sky_cfg.get(key,  default))
        assert val.unit != u.dimensionless_unscaled, (
            f'`{key}` should carry a unit.'
        )
        return val

    @classmethod
    def from_yaml_dict(
        cls,
        sky_cfg: dict,
    ):
        CENTER_RA_KEY = 'ra'
        CENTER_DEC_KEY = 'dec'

        return SkyCenter(
            cls._get_quantity(sky_cfg, CENTER_RA_KEY, RA_DEFAULT),
            cls._get_quantity(sky_cfg, CENTER_DEC_KEY, DEC_DEFAULT)
        )

    @classmethod
    def from_config_parser(cls, sky_cfg: ConfigParser):
        CENTER_RA_KEY = 'image center ra'
        CENTER_DEC_KEY = 'image center dec'

        return SkyCenter(
            cls._get_quantity(sky_cfg, CENTER_RA_KEY, RA_DEFAULT),
            cls._get_quantity(sky_cfg, CENTER_DEC_KEY, DEC_DEFAULT)
        )
