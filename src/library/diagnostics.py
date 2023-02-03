import numpy as np
import pickle
from matplotlib.colors import LogNorm

import nifty8 as ift

from src.library.utils import save_rgb_image_to_fits


def uncertainty_weighted_residual_image_from_file(sl_path_base, ground_truth_path, op=None,
                                                  output_dir_base=None):
    #FIXME: weighted residual in data space
    mpi_master = ift.utilities.get_MPI_params()[3]
    sl = ift.ResidualSampleList.load(sl_path_base)
    mean, var = sl.sample_stat(op)
    with open(ground_truth_path, "rb") as f:
         d = pickle.load(f)
    wgt_res = np.abs((mean-d).val)/np.sqrt(var.val)
    wgt_res = ift.Field.from_raw(domain=mean.domain, arr=wgt_res)
    if mpi_master and output_dir_base is not None:
        with open(f'{output_dir_base}.pkl', 'wb') as file:
            pickle.dump(wgt_res, file)
        save_rgb_image_to_fits(wgt_res, output_dir_base,
                               overwrite=True, MPI_master=mpi_master)
        p = ift.Plot()
        p.add(wgt_res, title= "Uncertainty weighted residual", norm=LogNorm())
        p.output(name=f'{output_dir_base}.png')
    return wgt_res

