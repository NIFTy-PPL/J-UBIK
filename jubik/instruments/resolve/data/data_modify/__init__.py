from .pipeline import modify_observation

try:
    from .classic_observation import convert_to_classic_observation
except ImportError:
    pass

from .polarization import *
from .autocorrelations import *
from .flagging import *
from .frequency import *
from .precision import *
from .visibility_subset import *
from .time import *
from .weights import *
