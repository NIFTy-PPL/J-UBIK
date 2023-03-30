import os
import pickle

import nifty8 as ift
import numpy as np

from .erosita_observation import ErositaObservation
from .sky_models import SkyModel
from .utils import get_cfg, create_output_directory, generate_mock_setup


def load_erosita_data(config_filepath, output_directory, diagnostics_directory, response_dict):
    cfg = get_cfg(config_filepath)

    # Mock info
    mock_run = cfg['mock']
    load_mock_data = cfg['load_mock_data']

    # Load file location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_ids = tel_info['tm_ids']

    # Load sky model
    sky_model = SkyModel(config_filepath)
    sky_dict = sky_model.create_sky_model()

    # Load mock position
    if mock_run:
        if sky_model.priors['point_sources'] is None:
            try:
                sky_model.priors['point_source'] = sky_model.config['point_source_defaults']
                mock_sky_dict = sky_model.create_sky_model()
                mock_sky_position = ift.from_random(mock_sky_dict['sky'].domain)
                sky_model.priors['point_sources'] = None
            except:
                raise ValueError(
                    'Not able to create point source model for mock data. Creating mock data'
                    'for diffuse only. Please check point_source_defaults in config!')
        else:
            mock_sky_position = ift.from_random(sky_dict['sky'].domain)

    # Prepare output dictionaries
    data_dict = {}
    masked_data_dict = {}

    for tm_id in tm_ids:
        tm_directory = create_output_directory(os.path.join(diagnostics_directory, f'tm{tm_id}'))
        output_filename = f'{tm_id}_' + file_info['output']

        tm_key = f'tm_{tm_id}'
        response_subdict = response_dict[tm_key]

        # Load mask
        mask = response_subdict[f'mask']

        if mock_run:
            print(f"Loading mock data for telescope module {tm_id}.")
            if load_mock_data:
                # FIXME: name of output folder for diagnostics into config
                # FIXME: Put Mockdata to a better place
                with open(diagnostics_directory + f'/tm{tm_id}_mock_sky_data.pkl', "rb") as f:
                    mock_data = pickle.load(f)
                data_dict[tm_key] = mock_data
            else:
                # Load response
                conv_op = response_subdict[f'convolution_op']
                exposure_field = response_subdict[f'exposure_field']
                mock_data_dict = generate_mock_setup(sky_model, conv_op,
                                                     mock_sky_position,
                                                     exposure_field,
                                                     sky_model.pad, tm_id,
                                                     output_directory=output_directory)
                mock_data = mock_data_dict['mock_data_sky']
                data_dict[tm_key] = mock_data

            # Mask mock data
            masked_data = mask(mock_data)
            masked_data_dict[tm_key] = masked_data

        else:
            observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)
            data = observation_instance.load_fits_data(output_filename)[0].data
            data = np.array(data, dtype=int)
            data = ift.makeField(sky_model.position_space, data)
            data_dict[tm_key] = data
            with open(tm_directory + f"/tm{tm_id}_data.pkl", "wb") as f:
                pickle.dump(data, f)
            masked_data = mask(data)
            masked_data_dict[tm_key] = masked_data

        # Print Exposure norm
        # norm = xu.get_norm(exposure, data)
        # print(norm)

    return data_dict, masked_data_dict
