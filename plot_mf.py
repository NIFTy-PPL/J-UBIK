import nifty7 as ift
import numpy as np
from lib.output import plot_slices


if True:
        dct = np.load('varinf_reconstruction.npy', allow_pickle = True).item()
        for  name in dct:
                plot_slices(dct[name],f'var_{name}.png', True)
        residual = ift.abs(dct['signal_response']- dct['data'])
        plot_slices(residual, 'residual.png', True)
# else:
#         dct = np.load('map_reconstruction.npy', allow_pickle = True).item()
#         for  name in dct:
#                 plot_result(dct[name],f'map_{name}.png')
