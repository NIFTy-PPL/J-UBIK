from matplotlib.colors import LogNorm

import xubik0 as xu
import nifty8 as ift

if __name__ == '__main__':
    n_samples = 6
    seed = 96

    path_to_config = 'eROSITA_config.yaml'
    priors_directory = 'priors/'
    cfg = xu.get_cfg(path_to_config)

    path_to_response = True  # decides whether to plot signal response
    if path_to_response is not None:
        resp_dict = xu.load_erosita_response(path_to_config, priors_directory)

    sky_dict = xu.SkyModel(path_to_config).create_sky_model()
    sky, power_spectrum = sky_dict['sky'], sky_dict['pspec']

    if 'point_sources' in sky_dict.keys():
        only_diffuse = False
        point_sources = sky_dict['point_sources']
        diffuse = sky_dict['diffuse']

    # Loads random seed for mock positions
    ift.random.push_sseq_from_seed(seed)
    positions = []
    for sample in range(n_samples):
        positions.append(ift.from_random(sky.domain))

    if 'point_sources' in sky_dict.keys():
        p1 = ift.Plot()
        p2 = ift.Plot()
        p4 = ift.Plot()
        p5 = ift.Plot()

    p3 = ift.Plot()
    p6 = ift.Plot()

    for pos in positions:
        if 'point_sources' in sky_dict.keys():
            p1.add(point_sources.force(pos), n_samples=n_samples, norm=LogNorm())
            p2.add(diffuse.force(pos), n_samples=n_samples, norm=LogNorm())
        p3.add(sky(pos), n_samples=n_samples, norm=LogNorm())

    p1.output(name=priors_directory + 'priors_point_sources.png', title='Point sources priors')
    p2.output(name=priors_directory + 'priors_diffuse.png', title='Diffuse priors')
    p3.output(name=priors_directory + 'priors_sky.png', title='Sky priors')
    print(f'Prior signal saved in {priors_directory}.')

    if path_to_response is not None:  # FIXME: when R will be pickled, load from file
        base_filename = 'sr{}_priors_{}.png'
        tm_ids = cfg['telescope']['tm_ids']

        for tm_id in tm_ids:
            tm_key = f'tm_{tm_id}'

            R = resp_dict[tm_key]['mask'].adjoint @ resp_dict[tm_key]['R']
            if 'point_sources' in sky_dict.keys():
                point_sources_response = R @ point_sources
                diffuse_response = R @ diffuse
            sky_response = R @ sky

            for pos in positions:
                if 'point_sources' in sky_dict.keys():
                    p4.add(point_sources_response.force(pos),
                           title=f'Point sources response tm {tm_id}',
                           norm=LogNorm())
                    p5.add(diffuse_response.force(pos), title=f'Diffuse response tm {tm_id}',
                           norm=LogNorm())
                p6.add(sky_response(pos), title=f'Sky response tm{tm_id}', norm=LogNorm())

            res_path = priors_directory + f'tm{tm_id}/'
            filename = res_path + base_filename
            if 'point_sources' in sky_dict.keys():
                p4.output(name=filename.format(tm_id, 'point_sources'),
                          title='Point sources signal response')
                p5.output(name=filename.format(tm_id, 'diffuse'), title='Diffuse signal response')
            p6.output(name=filename.format('sky'), title='Diffuse signal response')
            print(f'Signal response for tm {tm_id} saved as {filename.format(tm_id, "XX")}.')
