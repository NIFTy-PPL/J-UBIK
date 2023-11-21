import jubik0 as ju
from jax import random

if __name__ == '__main__':
    n_samples = 6
    seed = 96

    key = random.PRNGKey(seed)
    path_to_config = 'eROSITA_config.yaml'
    priors_directory = 'jubix_priors/'

    path_to_response = False # decides whether to plot signal response

    kwgs = {'n_cols': 3}
    ju.plot_erosita_priors(key, n_samples, path_to_config, path_to_response,
                           priors_directory, plotting_kwargs=kwgs)
