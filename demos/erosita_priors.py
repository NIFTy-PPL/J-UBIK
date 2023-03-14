from matplotlib.colors import LogNorm

import xubik0 as xu
import nifty8 as ift

if __name__ == '__main__':
    n_samples = 6
    seed = 96

    path_to_config = 'eROSITA_config.yaml'
    path_to_response = None

    sky_dict = xu.SkyModel(path_to_config).create_sky_model()
    sky, power_spectrum = sky_dict['sky'], sky_dict['pspec']

    only_diffuse = True
    if 'point_sources' in sky_dict.keys():
        only_diffuse = False
        point_sources = sky_dict['point_sources']
        diffuse = sky_dict['diffuse']

    ift.random.push_sseq_from_seed(seed)

    if not only_diffuse:
        ift.plot_priorsamples(point_sources, n_samples=n_samples, common_colorbar=False,
                              norm=LogNorm(), nx=3)
        ift.plot_priorsamples(diffuse, n_samples=n_samples, common_colorbar=False, norm=LogNorm(),
                              nx=3)
    ift.plot_priorsamples(sky, n_samples=n_samples, common_colorbar=False, norm=LogNorm(), nx=3)

    if path_to_response is not None:
        pass  # TODO: plot signal response
