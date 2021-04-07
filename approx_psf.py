import nifty7 as ift
import numpy as np
from psf_likelihood import makeModularModel

def slice(field, index, picdom):
    slc = field.val[index]
    slc_fld = ift.Field.from_raw(picdom, slc)
    return slc_fld

psf_trainset = np.load('strainset_psf.npy', allow_pickle=True).item()
pos, sr, data = makeModularModel(psf_trainset)

plt = ift.Plot()
for i in range(9):
    plt.add(ift.log10(slice(ift.abs(sr(pos)-data), i, ift.RGSpace([256,256]))))
plt.output(nx=3,ny=3, xsize=30, ysize=30, name='res.png' )
np.save('trained_modular.npy', pos)
