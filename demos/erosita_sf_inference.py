import argparse
import os
import sys
import nifty8 as ift
import xubik0 as xu
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print(sys.path)
print
from src.library.erosita_observation import ErositaObservation

if __name__ == "__main__":
    cfg = xu.get_cfg("demos/eROSITA_config.yaml")
    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)

    # Grid Info

    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info

    tel_info = cfg['telescope']
    tm_id = tel_info['tm_id']


    log = 'Output file {} already exists and is not regenerated. If the observations parameters shall be changed ' \
          'please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=tel_info['rebin'],
                                                    size=npix, pattern=tel_info['pattern'],
                                                    telid=tm_id)
    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    observation_instance = ErositaObservation(output_filename, output_filename, obs_path)

    # Exposure
    if not os.path.exists(os.path.join(obs_path, exposure_filename)):
        observation_instance.get_exposure_maps(output_filename, e_min, e_max, mergedmaps=exposure_filename)

    else:
        print(log.format(os.path.join(obs_path, output_filename)))
    # Plotting
    plot_info = cfg['plotting']
    if plot_info['enabled']:
        observation_instance.plot_fits_data(output_filename,
                                            os.path.splitext(output_filename)[0],
                                            slice=tuple(plot_info['slice']),
                                            dpi=plot_info['dpi'])
        observation_instance.plot_fits_data(exposure_filename,
                                            f'{os.path.splitext(exposure_filename)[0]}.png',
                                            slice=tuple(plot_info['slice']),
                                            dpi=plot_info['dpi'])

    data = observation_instance.load_fits_data(output_filename)[0].data
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data

    data_space = ift.RGSpace(data.shape, distances=0.1) # fixme: replace by signal.target
    data = ift.makeField(data_space, data) # todo: check nifty plotting. data.T?

    exposure = ift.makeField(data_space, exposure)
    exposure_op = ift.makeOp(exposure)
    mask = xu.get_mask_operator(exposure)
    masked_data = mask(data)

    R = mask @ exposure_op

    # Set up likelihood
    # log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ signal

    p = ift.Plot()
    p.add(data, norm=colors.SymLogNorm(linthresh=10e-5))
    p.add(exposure)
    p.output(nx=2)


