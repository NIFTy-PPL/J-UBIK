import numpy as np
import pickle

import nifty8 as ift
import xubik0 as xu


if __name__ == "__main__":
    """This is the postprocessing pipeline for the eROSITA reconstruction:
    The uncertainty weighted residual (UWR) as well as their distribution 
    and the uncertainty weighted mean (UWM) are calculated. For mock data
    additionally the UWR in signal space is calculated."""



    # Paths -Set by user
    reconstruction_path = "../rec_weak_points/" #FIXME filepath
    config_filename = "eROSITA_config.yaml"
    sl_path_base = reconstruction_path + "pickle/last" # NIFTy dependency
    data_path = reconstruction_path + "diagnostics/mock_data_sky.pkl"
    response_path = "" # FIXME response operator shall be loaded from path
    output_dir_base = reconstruction_path + "diagnostics"
    # Ground Truth path Only needed for mock run
    ground_truth_path = reconstruction_path + "diagnostics/mock_sky.pkl"


    # Config
    cfg = xu.get_cfg(reconstruction_path + config_filename)
    mock_run = cfg['mock']
    mock_psf = cfg['mock_psf']
    file_info = cfg['files']
    output_filename = file_info['output']
    obs_path = file_info['obs_path']
    exposure_filename = file_info['exposure']


    # Operators
    # Sky
    sky_model = xu.SkyModel(reconstruction_path + config_filename)
    sky_dict = sky_model.create_sky_model()
    # Repsonse # FIXME run
    observation_instance = xu.ErositaObservation(output_filename, output_filename, obs_path)
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data
    exposure_field = ift.makeField(sky_model.position_space, exposure)
    padded_exposure_field = sky_model.pad(exposure_field)
    exposure_op = ift.makeOp(padded_exposure_field)
    mask = xu.get_mask_operator(exposure_field)
    try:
        with open(response_path, "rb") as f:
            R = pickle.load(f)
    except:
        print('Not able to load response from file. Generating response from config ...')
        if cfg['psf']['method'] in ['MSC', 'LIN']:
            center = observation_instance.get_center_coordinates(output_filename)
            psf_file = xu.eROSITA_PSF(cfg["files"]["psf_path"])  # FIXME: load from config

            dom = sky_model.extended_space
            center = tuple(0.5*ss*dd for ss, dd in zip(dom.shape, dom.distances))

            energy = cfg['psf']['energy']
            conv_op = psf_file.make_psf_op(energy, center, sky_model.extended_space,
                                        conv_method=cfg['psf']['method'],
                                        conv_params=cfg['psf'])

        elif cfg['psf']['method'] == 'invariant':
            if mock_psf:
                conv_op = xu.get_gaussian_psf(sky_dict['sky'], var=cfg['psf']['gauss_var'])
            else:
                center = observation_instance.get_center_coordinates(output_filename)
                psf_file = xu.eROSITA_PSF(cfg["files"]["psf_path"])
                psf_function = psf_file.psf_func_on_domain('3000', center, sky_model.extended_space)
                psf_kernel = psf_function(*center)
                psf_kernel = ift.makeField(sky_model.extended_space, np.array(psf_kernel))
                conv_op = xu.get_fft_psf_op(psf_kernel, sky_dict['sky'])
        else:
            raise NotImplementedError

    R = mask @ sky_model.pad.adjoint @ exposure_op @ conv_op
    if mock_run:
        xu.signal_space_uwr_from_file(sl_path_base=sl_path_base, ground_truth_path=ground_truth_path,
                                      sky_op=sky_dict['sky'], output_dir_base=output_dir_base+'/signal_space_uwr')
    xu.data_space_uwr_from_file(sl_path_base=sl_path_base, data_path=data_path, sky_op=sky_dict['sky'],
                                response_op=R, mask_op=mask, output_dir_base=output_dir_base+'/data_space_uwr')
    xu.signal_space_uwm_from_file(sl_path_base=sl_path_base, sky_op=sky_dict['sky'],
                                  output_dir_base=output_dir_base+'/uwm')
    xu.weighted_residual_distribution(sl_path_base=sl_path_base, data_path=data_path, sky_op=sky_dict['sky'],
                                      response_op=R, mask_op=mask,
                                      output_dir_base=output_dir_base+'/res_distribution')
