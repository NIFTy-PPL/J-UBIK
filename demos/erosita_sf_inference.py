import argparse
import os.path

import nifty8 as ift
import xubik0 as xu
from src.library.erosita_observation import ErositaObservation
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("obs_path", type=str, nargs='?', default="../data/LMC_SN1987A/")
parser.add_argument("plotting", type=bool, nargs='?', default=False)
args = parser.parse_args()

if __name__ == "__main__":
    emin = 1.0
    emax = 2.3

    obs_path = args.obs_path  # Folder that gets mounted to the docker
    input_filenames = ['fm00_700203_020_EventList_c001.fits']
    output_filename = "processed_LMC_SN1987A_data.fits"
    exposure_filename = "expmap_LMC_SN1987A.fits"
    observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)

    log = 'Output file {} already exists and is not regenerated. If the observations parameters shall be changed ' \
          'please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=emin, emax=emax, image=True, rebin=80, size=3240, pattern=15,
                                                    telid=1)
    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    observation_instance = ErositaObservation(output_filename, output_filename, obs_path)

    # Exposure
    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation_instance.get_exposure_maps(output_filename, emin, emax, mergedmaps=exposure_filename)

    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    if args.plotting:
        observation_instance.plot_fits_data(output_filename, "combined_out_08_1_no_imm", slice=(1100, 2000, 800, 2000),
                                            dpi=800)
        observation_instance.plot_fits_data(exposure_filename, "exposure_LMC_SN1987A.png",
                                            slice=(1100, 2000, 800, 2000), dpi=800)

    data = observation_instance.load_fits_data(output_filename)[0].data
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data

    data_space = ift.RGSpace(data.shape, distances=0.1) # fixme: replace by signal.target
    data = ift.makeField(data_space, data) # todo: check nifty plotting. data.T?
    exposure = ift.makeOp(ift.makeField(data_space, exposure))

    mask = xu.get_mask_operator(exposure)
    masked_data = mask(data)

    R = mask @ exposure

    # Set up likelihood
    log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ signal

    # p = ift.Plot()
    # p.add(data, norm=colors.SymLogNorm(linthresh=10e-5))
    # p.add(exposure)
    # p.output(nx=2)


