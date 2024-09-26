# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %

from .version import __version__
from .utils import (save_to_pickle, load_from_pickle, get_config, save_to_yaml,
                    copy_config, create_output_directory,
                    coord_center, get_stats, add_masked_array)
from .plot import (plot_result, plot_histograms,
                   plot_sample_averaged_log_2d_histogram, plot_rgb,
                   _clip, _non_zero_log, _norm_rgb_plot)
from .sugar_plot import (plot_pspec, plot_sample_and_stats,
                         plot_sample_and_stats, plot_erosita_priors,
                         plot_2d_gt_vs_rec_histogram,
                         plot_noise_weighted_residuals,
                         plot_uncertainty_weighted_residuals)

from .instruments.erosita.erosita_observation import ErositaObservation
from .instruments.chandra.chandra_observation import ChandraObservationInformation
from .instruments.chandra.chandra_psf import (get_radec_from_xy,
                                              get_psfpatches,
                                              get_synth_pointsource)
from .instruments.erosita.erosita_psf import eROSITA_PSF
from .instruments.chandra.chandra_data import (generate_chandra_data,
                                               create_chandra_data_from_config)
from .instruments.chandra.chandra_response import build_chandra_response_from_config
from .instruments.chandra.chandra_likelihood import generate_chandra_likelihood_from_config
from .sky_models import SkyModel, MappedModel, GeneralModel, _apply_slope
from .response import build_exposure_function, build_readout_function
from .instruments.erosita.erosita_response import (build_callable_from_exposure_file,
                                                   build_erosita_psf,
                                                   build_erosita_response_from_config,
                                                   load_erosita_response,
                                                   calculate_erosita_effective_area)
from .instruments.erosita.erosita_data import (generate_erosita_data_from_config,
                                               mask_erosita_data_from_disk)
from .instruments.erosita.erosita_psf import eROSITA_PSF
from .instruments.jwst.jwst_response import build_jwst_response
from .instruments.jwst.reconstruction_grid import Grid

from .data import (create_mock_data, load_masked_data_from_config,
                   load_mock_position_from_config, Domain)
from .likelihood import get_n_constrained_dof
from .instruments.erosita.erosita_likelihood import generate_erosita_likelihood_from_config
from .diagnostics import calculate_nwr, calculate_uwr
from .convolve import linpatch_convolve, convolve, slice_patches
from .minimization_parser import MinimizationParser
from .hashcollector import save_local_packages_hashes_to_txt
