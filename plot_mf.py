import nifty7 as ift
import numpy as np
from lib.output import plot_slices

mdata = np.load('mdata.npy', allow_pickle=True).item()
if True:
        dct = np.load('varinf_reconstruction.npy', allow_pickle = True).item()
        # for  name in dct:
                # plot_slices(dct[name],f'var_{name}.png', True)
        # residual = ift.abs(dct['signal_response']- dct['data'])
        # residual = ift.abs(dct['signal_response']- mdata)

        data = dct['data']
        mask = np.zeros(data.shape)
        mask[data.val == 0] = 1
        mask = ift.Field.from_raw(data.domain, mask)
        mask = ift.MaskOperator(mask)

        sr = dct['signal_response']
        sr_datadom = mask.adjoint(mask(sr))
        residual = ift.abs(sr_datadom - data)
        residual[residual.val]
        plot_slices(residual, 'residual.png', True)
        plot_slices(sr, 'sr.png', True)
        plot_slices(sr_datadom, 'srdata.png', True)

# else:
#         dct = np.load('map_reconstruction.npy', allow_pickle = True).item()
#         for  name in dct:
#                 plot_result(dct[name],f'map_{name}.png')
