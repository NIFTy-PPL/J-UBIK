import os.path

import nifty8 as ift

import xubik0 as xu
from eROSITA_sky import ErositaSky
import matplotlib as mpl
import matplotlib.pyplot as plt


def configure_nice_plot_settings():
    nice_fonts = {'text.usetex': True, 'pgf.texsystem': 'pdflatex', 'axes.unicode_minus': False, 'font.family': 'serif',
                  'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 11, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11}
    mpl.rcParams.update(nice_fonts)
    plt.style.use('seaborn-paper')


def create_output_directory():
    output_directory = os.path.join(os.path.curdir, "demo_results")
    return output_directory


def im_plotter(field, filename, extension, x_label, y_label, figsize=None, dpi=None, **kwargs):
    """
    # FIXME: Finish docstring
    Parameters
    ----------
    figsize
    y_label
    x_label
    field
    filename
    extension : the field-specific name e.g. data

    Returns
    -------

    """

    configure_nice_plot_settings()
    filename_specific = filename.format(extension)
    plt.figure(figsize=figsize, dpi=dpi)
    # plt.title('Probability Distribution of Viral Load for Different Ages')
    plt.imshow(field.val, origin="lower", **kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    # cbar.yticklabels(ticklabs, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename_specific)
    plt.close()
    print("Saved results as '{}'.".format(filename_specific))


if __name__ == '__main__':
    # Load the data
    obs_path = "../data/"  # Folder that gets mounted to the docker
    filename = "combined_out_08_1_imm.fits"
    input_filename = ['LMC_SN1987A/fm00_700203_020_EventList_c001.fits',
                      'LMC_SN1987A/fm00_700204_020_EventList_c001.fits',
                      'LMC_SN1987A/fm00_700204_020_EventList_c001.fits']

    observation_instance = xu.ErositaObservation(input_filename, filename, obs_path)  # load an observation object
    # observation = observation_instance.get_data(emin=0.7, emax=1.0, image=True, rebin=80, size=3240, pattern=15,
    #                                             gti='GTI') # combine 3 datasets into an image saved in filename

    # observation_instance_2 = ErositaObservation(filename, filename) # load a new observation from the merged image
    # observation_instance_2.get_exposure_maps(filename, 0.7, 1.0, mergedmaps="expmap_combined.fits") # generate expmaps

    observation = observation_instance.load_fits_data(filename)
    data = observation[0].data
    image_filename = "combined_out_08_1_imm.png"
    # observation_instance.plot_fits_data(filename, image_filename, slice=(1100, 2000, 800, 2000), dpi=800) # plot data
    data = data[800:2000, 1100:2000]  # slice the data
    expmap = observation_instance.load_fits_data("expmap_combined.fits")  # load expmaps
    # observation_instance.plot_fits_data("expmap_combined.fits", "prova.png",  slice=(1100, 2000, 800, 2000),
    # dpi=800)  # plot expmaps
    expmap = expmap[0].data[800:2000, 1100:2000]  # slice expmap as data

    # Load sky model
    erositaModel = ErositaSky(data, expmap, "eROSITA_config.yaml")
    sky_field = erositaModel.full_sky(ift.from_random(erositaModel.full_sky.domain))

    # Set up likelihood
    log_likelihood = ift.PoissonianEnergy(erositaModel.masked_data) @ erositaModel.signal

    # Load minimization config
    minimization_config = xu.get_cfg('eROSITA_config.yaml')['minimization']

    # Minimizers
    ic_newton = ift.AbsDeltaEnergyController(**minimization_config['ic_newton'])
    ic_sampling = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling'])
    ic_sampling_nl = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling_nl'])
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Create a plots directory
    output_directory = create_output_directory()
    results_directory = os.path.join(create_output_directory(), "results")
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    # Load posterior samples
    samples = ift.ResidualSampleList.load(output_directory + "/pickle/last")

    padder = erositaModel.pad.adjoint
    sky_mean, _ = samples.sample_stat(erositaModel.sky)
    ps_mean, _ = samples.sample_stat(erositaModel.point_sources)
    diffuse_mean, _ = samples.sample_stat(erositaModel.diffuse_component)

    im_plotter(sky_mean, os.path.join(results_directory, "reconstruction_mean.pdf"), None, None, None, norm=mpl.colors.SymLogNorm(linthresh=10e-1))
    im_plotter(padder(ps_mean), os.path.join(results_directory, "ps_mean.pdf"), None, None, None, norm=mpl.colors.SymLogNorm(linthresh=10e-1))
    im_plotter(padder(diffuse_mean), os.path.join(results_directory, "diffuse_mean.pdf"), None, None, None, norm=mpl.colors.SymLogNorm(linthresh=10e-1))






