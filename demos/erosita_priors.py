from matplotlib.colors import LogNorm

import xubik0 as xu

if __name__ == '__main__':
    n_samples = 6
    seed = 96

    path_to_config = 'eROSITA_config.yaml'
    priors_directory = 'priors/'

    path_to_response = True  # decides whether to plot signal response

    kwgs = {'norm': LogNorm(), 'nx': 3}
    xu.plot_erosita_priors(seed, n_samples, path_to_config, path_to_response, priors_directory,
                           plotting_kwargs=kwgs)
