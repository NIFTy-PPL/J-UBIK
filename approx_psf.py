import nifty7 as ift
import numpy as np
from psf_likelihood import makeModularModel

def slice(field, index, picdom):
    slc = field.val[index]
    slc_fld = ift.Field.from_raw(picdom, slc)
    return slc_fld

psf_trainset = np.load('trainset_psf.npy', allow_pickle=True).item()
test = makeModularModel(psf_trainset)

