import os
import sys
import numpy as np
import unittest

import nifty8 as ift

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src.library.utils import generate_mock_data, get_cfg


def test(config_filename):
    try:
        config_path = config_filename
        cfg = get_cfg(config_path)
    except:
        config_path = 'demos/' + config_filename
        cfg = get_cfg(config_path)
    sky_model = xu.SkyModel(config_path)
    exposures = [None, ift.makeField(sky_model.position_space,
                                     np.random.choice([0, 1], sky_model.position_space.shape))]
    padders = [None, sky_model.pad]
    ift.logger.info(sky_model.pad.domain)
    psf_kernels = [None, ift.makeField(sky_model.extended_space,
                                       np.random.choice([0.0, 1.0], sky_model.extended_space.shape))]
    vars = [None, 5]
    output_directories = [None, 'test_mock_data_generation/']
    for exposure in exposures:
        for padder in padders:
            for psf_kernel in psf_kernels:
                for var in vars:
                    for output_directory in output_directories:
                        try:
                            generate_mock_data(sky_model=sky_model,
                                               exposure=exposure,
                                               pad=padder,
                                               psf_kernel=psf_kernel,
                                               var=var,
                                               output_directory=output_directory)
                        except ValueError as v:
                            if var is None and psf_kernel is None:
                                pass
                            elif padder is None and sky_model.position_space != sky_model.extended_space:
                                pass
                            else:
                                print(f'Unexpected Value Error {v} for combination: {exposure}, {padder}, {psf_kernel},'
                                      f'{var}, {output_directory}')
                        except Exception as e:
                            print(f'Unexpected Exception {e} for combination: {exposure}, {padder}, {psf_kernel},'
                                   f'{var}, {output_directory}')


cfg_filename = 'eROSITA_config.yaml'
test(cfg_filename)

