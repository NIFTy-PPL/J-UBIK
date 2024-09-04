import numpy as np

import nifty8 as ift
import jubik0 as ju

obs_info = ju.get_config("obs/obs.yaml")
cfg = ju.get_config("scripts/config.yaml")
grid = cfg["grid"]
npix_s = grid["npix_s"]
fov = grid["fov"]
numrays = 1e7

info = ju.ChandraObservationInformation(obs_info["obs4952"], **grid)
idxs = [(130, 130), (500, 500), (700, 700)]
sp = ift.RGSpace([npix_s, npix_s])

for i, idx_tupel in enumerate(idxs):
    point_source = ju.get_synth_pointsource(info, npix_s, idx_tupel, numrays)
    point_source = point_source[:, :, 0]
    point_source = ift.makeField(sp, point_source)
    if i == 0:
        ps_sum = point_source
    else:
        ps_sum = ps_sum + point_source

test_pic = np.zeros([1024, 1024])
test_pic[50, 50] = 1
test_pic[500, 500] = 1
test_pic[700, 700] = 1
test = ift.makeField(sp, test_pic)
np.save("synth_data.npy", ps_sum)
