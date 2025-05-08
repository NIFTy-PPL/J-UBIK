from dataclasses import dataclass

from astropy import units as u


@dataclass
class SkyMetaInformation:
    grid_extension: tuple[int, int]
    unit: u.Unit
    dvol: u.Quantity
