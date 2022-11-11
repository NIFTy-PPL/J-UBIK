import os.path

import nifty8 as ift

import xubik0 as xu
from eROSITA_sky import ErositaSky


# TODO: polish and move these functions to utils
def _append_key(s, key):
    if key == "":
        return s
    return f"{s} ({key})"


def _plot_samples(filename, samples, plotting_kwargs):
    samples = list(samples)

    if isinstance(samples[0].domain, ift.DomainTuple):
        samples = [ift.MultiField.from_dict({"": ss}) for ss in samples]
        # if ground_truth is not None:
        #     ground_truth = ift.MultiField.from_dict({"": ground_truth})
    if not all(isinstance(ss, ift.MultiField) for ss in samples):
        raise TypeError
    keys = samples[0].keys()

    p = ift.Plot()
    for kk in keys:
        single_samples = [ss[kk] for ss in samples]

        if ift.plot.plottable2D(samples[0][kk]):
            # if ground_truth is not None:
            # p.add(ground_truth[kk], title=_append_key("Ground truth", kk),
            #       **plotting_kwargs)
            # p.add(None)
            for ii, ss in enumerate(single_samples):
                # if (ground_truth is None and ii == 16) or (ground_truth is not None and ii == 14):
                #     break
                p.add(ss, title=_append_key(f"Sample {ii}", kk), **plotting_kwargs)
        else:
            n = len(samples)
            alpha = n * [0.5]
            color = n * ["maroon"]
            label = None
            # if ground_truth is not None:
            #     single_samples = [ground_truth[kk]] + single_samples
            #     alpha = [1.] + alpha
            #     color = ["green"] + color
            #     label = ["Ground truth", "Samples"] + (n-1)*[None]
            p.add(single_samples, color=color, alpha=alpha, label=label,
                  title=_append_key("Samples", kk), **plotting_kwargs)
    p.output(name=filename)


def _plot_stats(filename, op, sl, plotting_kwargs):
    try:
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    mean, var = sl.sample_stat(op)
    p = ift.Plot()
    # if op is not None: TODO: add Ground Truth plotting capabilities
    #     p.add(op, title="Ground truth", **plotting_kwargs)
    p.add(mean, title="Mean", **plotting_kwargs)
    p.add(var.sqrt(), title="Standard deviation")
    p.output(name=filename, ny=2)
    # print("Output saved as {}.".format(filename))


def plot_sample_and_stats(output_directory, operators_dict, sample_list, iterator, plotting_kwargs):
    for key in operators_dict:
        op = operators_dict[key]
        results_path = os.path.join(output_directory, key)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        filename = os.path.join(output_directory, key, "stats_{}.png".format(iterator))
        filename_samples = os.path.join(output_directory, key, "samples_{}.png".format(iterator))

        _plot_stats(filename, op, sample_list, plotting_kwargs)
        _plot_samples(filename_samples, sample_list.iterator(op), plotting_kwargs)


def create_output_directory():
    output_directory = os.path.join(os.path.curdir, "demo_results")
    return output_directory


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

    # Prepare results
    operators_to_plot = {'reconstruction': erositaModel.sky, 'full_sky': erositaModel.full_sky}

    # Create an output directory
    output_directory = create_output_directory()

    import matplotlib.colors as colors

    plot = lambda x, y: plot_sample_and_stats(output_directory, operators_to_plot, x, y,
                                              plotting_kwargs={'norm': colors.SymLogNorm(linthresh=10e-1)})

    # MGVI
    ift.optimize_kl(log_likelihood, minimization_config['total_iterations'], minimization_config['n_samples'],
                    minimizer, ic_sampling, None, export_operator_outputs=operators_to_plot,
                    output_directory=output_directory, inspect_callback=plot)

    # Plot prior sampleR
    # p = ift.Plot()
    # import matplotlib.colors as colors
    # p.add(sky_field, norm=colors.SymLogNorm(linthresh=10e-1))
    # output_name = "erosita_priors.png"
    # p.output(name=output_name)
    # print("Output saved as {}.".format(output_name))
