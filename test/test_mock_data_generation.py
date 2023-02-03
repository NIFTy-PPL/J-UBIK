import numpy as np
import pytest

import nifty8 as ift
import xubik0 as xu


pmp = pytest.mark.parametrize

cfg_filename = 'eROSITA_config.yaml'
config_path = 'demos/' + cfg_filename  # FIXME cfg in testdir!!
cfg = xu.get_cfg(config_path)
sky_model = xu.SkyModel(config_path)

psf_kernel = ift.makeField(sky_model.extended_space,
                           np.random.choice([0.0, 1.0], sky_model.extended_space.shape))
exposure = ift.makeField(sky_model.position_space,
                         np.random.choice([0, 1], sky_model.position_space.shape))
output_directory = 'test_mock_data_generation/'

@pmp('gauss_var', [None, 5])
@pmp('psf_kernel', [None, psf_kernel])
@pmp('exposure', [None, exposure])
@pmp('padder', [sky_model.pad])
def test(psf_kernel, gauss_var, exposure, padder):
    if gauss_var is None and psf_kernel is None:
        pass
    else:
        ift.logger.info(sky_model.pad.domain)
        xu.generate_mock_data(sky_model=sky_model,
                              exposure=exposure,
                              pad=padder,
                              psf_kernel=psf_kernel,
                              var=gauss_var,
                              output_directory=output_directory)
