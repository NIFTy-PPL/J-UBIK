#!/usr/bin/env
#

import xubik0 as xu
import nifty8 as ift
import numpy as np
from os.path import join, isfile
from functools import reduce
import time

with open('models/mf_sky_trans.py', 'r') as fd:
    exec(fd.read())
#cfg = xu.get_cfg("config_mf.yaml")
output_directory = "df_rec_old"
fname = "last"
fname = reduce(join, [output_directory, "pickle", fname])
if isfile(fname + ".mean.pickle"):
    rsl = ift.ResidualSampleList
    initial_position = rsl.load_mean(fname)
    signal_field_mean, signal_field_std = rsl.load(fname).sample_stat(signal)
    points_field_mean, points_field_std = rsl.load(fname).sample_stat(points)
    diffuse_field_mean, diffuse_field_std = rsl.load(fname).sample_stat(diffuse)
    mpi = rsl.MPI_master
else:
    sl = ift.SampleList.load(fname)
    myassert(sl.n_samples == 1)
    initial_position = sl.local_item(0)
    signal_field_mean, signal_field_std = sl.sample_stat(signal)
    points_field = sl.sample_stat(points)
    diffuse_field = sl.sample_stat(diffuse)
    mpi = sl.MPI_master
print(len(signal_field_mean.domain))
print(points_field_mean.domain)
print(diffuse_field_mean.domain)

binned_signal_field = xu.energy_binning(signal_field_mean, 3)
print(binned_signal_field)

xu.save_rgb_image_to_fits(fld = binned_signal_field, file_name = 'test', overwrite=True, mpi=mpi)

xu.plot_rgb_image(file_name_in='test', file_name_out='test_out.jpeg')
