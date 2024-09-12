import numpy as np
import pytest

import nifty8 as ift
import jubik0 as ju

config_path = 'test/mock_data_generation.yaml'
cfg = ju.get_config(config_path)
sky_model = ju.SkyModel(config_path)

psf_arr = np.random.choice([0.0, 1.0], sky_model.extended_space.shape)
psf_kernel = ift.makeField(sky_model.extended_space, psf_arr)
psf_op = ju.get_fft_psf_op(psf_kernel, sky_model.extended_space)
exposure_arr = np.random.choice([0, 1], sky_model.position_space.shape)
exposure = ift.makeField(sky_model.position_space, exposure_arr)
output_directory = 'test_mock_data_generation/'

pmp = pytest.mark.parametrize


@pmp('gauss_var', [None, 5])
@pmp('psf_kernel', [None, psf_kernel])
@pmp('exposure', [None, exposure])
@pmp('padder', [sky_model.pad])
def test_generate_mock_setup(psf_kernel, gauss_var, exposure, padder):
    if gauss_var is None and psf_kernel is None:
        pass
    else:
        ju.generate_mock_setup(sky_model=sky_model,
                              exposure=exposure,
                              pad=padder,
                              psf_op=psf_op,
                              output_directory=output_directory)