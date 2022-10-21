import os

from matplotlib import colors

from xubik.src.operators.erosita_observation import ErositaObservation
import nifty8 as ift
import matplotlib.pyplot as plt

if __name__ == "__main__":
    obs_path = "../../../data/" # Folder that gets mounted to the dockerr
    filename = "output_pm.fits"
    input_filename = "LMC_SN1987A/pm00_700161_020_EventList_c001.fits"
    output_filename = obs_path + os.path.splitext(input_filename)[0] + ".png"


    observation_instance = ErositaObservation(input_filename, filename)
    observation = observation_instance.load_data("output_pm.fits")
    data = observation[0].data
    plt.imshow(data.T, origin='lower', norm=colors.SymLogNorm(linthresh=1e-1))
    # plt.imshow(expmap, origin='lower', norm=colors.SymLogNorm(linthresh=8 * 10e-3))
    # plt.imshow(psfmap, origin='lower', norm=colors.SymLogNorm(linthresh=8 * 10e-8))
    # plt.plot
    plt.colorbar()
    plt.show()