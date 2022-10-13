import nifty8 as ift
import numpy as np


psf_file = np.load("data/npdata/psf_patches/obs4952_patches_v1.npy", allow_pickle=True).item()["psf_sim"]
new_list = []
for p in psf_file:
        arr = p.val_rw()
        co_x, co_y = np.unravel_index(np.argmax(arr), p.domain.shape)
        print(np.argmax(arr))
        tmp_psf_sim = np.roll(np.copy(arr), (-co_x, -co_y), axis=(0, 1))
        print(np.argmax(tmp_psf_sim))
        tmp_field = ift.Field.from_raw(p.domain, tmp_psf_sim)
        new_list.append(tmp_field)

dic = {"psf_sim" : new_list}
np.save("data/npdata/psf_patches/shifted.npy", dic)
