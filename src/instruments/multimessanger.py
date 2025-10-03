from ..color import Color
from .resolve.telescopes.alma import ALMA_RANGE

from enum import Enum


class InstrumentRanges(Enum):
    alma_range = ALMA_RANGE
