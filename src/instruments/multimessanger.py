from ..color import ColorRange
from .resolve.telescopes.alma import ALMA_RANGE

from enum import Enum


class InstrumentRanges(Enum):
    alma_range: ColorRange = ALMA_RANGE
