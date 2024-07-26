import nifty8.re as jft
import jax.numpy as jnp
from .reconstruction_grid import Grid
from astropy import units

from ..library.sky_models import SkyModel
from ..library.sky_colormix import ColorMix, Components, build_color_matrix


def build_sky_model(shape, dist, offset, fluctuations, extend_factor=1.5):
    assert len(shape) == 2

    cfm = jft.CorrelatedFieldMaker(prefix='reco')
    cfm.set_amplitude_total_offset(**offset)
    if 'non_parametric_kind' not in fluctuations:
        fluctuations['non_parametric_kind'] = 'power'
    cfm.add_fluctuations(
        [int(shp*extend_factor) for shp in shape], dist,
        **fluctuations)
    log_diffuse = cfm.finalize()

    # ext0, ext1 = [int(shp*extend_factor - shp)//2 for shp in shape]

    # def diffuse(x):
    #     return jnp.exp(log_diffuse(x)[ext0:-ext0, ext1:-ext1])

    ext0, ext1 = [int(shp*extend_factor - shp) for shp in shape]

    def diffuse(x):
        return jnp.exp(log_diffuse(x)[:-ext0, :-ext1])

    def full_diffuse(x):
        return jnp.exp(log_diffuse(x))

    return (jft.Model(diffuse, domain=log_diffuse.domain),
            jft.Model(full_diffuse, domain=log_diffuse.domain))


def build_components(
    prefix: str,
    shape: tuple[int],
    distances: tuple[int],
    padding_ratio: float,
    prior_config: dict
):

    def component(key, shape, distances, config):
        cfm = jft.CorrelatedFieldMaker(prefix=key)
        cfm.set_amplitude_total_offset(**config['offset'])
        cfm.add_fluctuations(shape, distances, **config['fluctuations'])
        return cfm.finalize()

    prefix = f'{prefix}_comp_'

    pad_shape = [int(s*padding_ratio) for s in shape]

    components = []
    for key, config in prior_config.items():
        components.append(
            component(f'{prefix}_{key}', pad_shape, distances, config))

    return Components(components, shape)


def build_colormix_components(
    prefix: str,
    colormix_config: dict,
    components_config: dict
):

    comps = build_components(
        prefix=prefix,
        shape=components_config['shape'],
        distances=components_config['distances'],
        padding_ratio=components_config['s_padding_ratio'],
        prior_config=components_config['prior'])

    color = build_color_matrix(
        prefix,
        comps.target,
        diagonal_prior=colormix_config['diagonal_prior'],
        off_diagonal_prior=colormix_config['off_diagonal_prior'])

    return ColorMix(comps, color)


def prior_samples_colormix_components(sky_model: ColorMix, n_samples=4):
    from jax import random
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    key = random.PRNGKey(42)
    N_comps = len(sky_model.components._comps)

    for _ in range(n_samples):
        key, rec_key = random.split(key, 2)
        x = jft.random_like(key, sky_model.domain)

        comps = sky_model.components(x)
        correlated_comps = sky_model(x)

        mat_mean = sky_model.color_matrix(x)
        print()
        print('Color Mixing Matrix')
        print(mat_mean)
        print()

        fig, axes = plt.subplots(N_comps, 2)
        for ax, cor_comps, comps in zip(axes, correlated_comps, comps):
            im0 = ax[0].imshow(cor_comps, origin='lower', norm=LogNorm())
            im1 = ax[1].imshow(jnp.exp(comps), origin='lower', norm=LogNorm())
            plt.colorbar(im0, ax=ax[0])
            plt.colorbar(im1, ax=ax[1])
            ax[0].set_title('Correlated Comps')
            ax[1].set_title('Comps')

        plt.show()


def build_sky_model_from_config(
        config: dict, reconstruction_grid: Grid, plot=False) -> jft.Model:

    if 'mean' in config['priors']:
        from charm_lensing.models.hybrid_model import build_hybrid_model
        from charm_lensing.spaces import Space

        model_cfg = dict(
            mean=config['priors']['mean'],
            perturbations=dict(
                ubik=dict(priors=config['priors'], grid=config['grid'],
                          energy_bin=config['grid']['energy_bin']))
        )
        space = Space(
            shape=reconstruction_grid.shape,
            distances=[
                d.to(units.arcsec).value for d in reconstruction_grid.distances],
            space_key='',
            extend_factor=config['grid'].get('s_padding_ratio', 1.0),
        )
        small_sky_model = sky_model = build_hybrid_model(
            space=space,
            model_key='light',
            model_cfg=model_cfg)

        alpha_tmp = sky_model.nonparametric()._sky_model.alpha_cf
        energy_cfg = sky_model.nonparametric(
        )._sky_model.config['grid']['energy_bin']
        def alpha(x): return alpha_tmp(x)[:sdim, :sdim]

    if config['priors']['diffuse'].get('colormix'):
        from copy import deepcopy
        energy_bins = config['grid'].get('edim')
        energy_cfg = config['grid'].get('energy_bin')
        diffuse_priors = config['priors']['diffuse']

        components_prior_config = {
            f'k{ii}': deepcopy(diffuse_priors['spatial']) for ii in range(energy_bins)}

        components_config = dict(
            shape=reconstruction_grid.shape,
            distances=[
                d.to(units.arcsec).value for d in reconstruction_grid.distances],
            s_padding_ratio=config['grid'].get('s_padding_ratio', 1.0),
            prior=components_prior_config,
        )

        small_sky_model = sky_model = build_colormix_components(
            'sky',
            colormix_config=diffuse_priors['colormix'],
            components_config=components_config)

        def alpha(x):
            return jnp.ones((10, 10))

        if plot:
            prior_samples_colormix_components(sky_model, 4)

    else:
        sky_model_new = SkyModel(config_file_path=config)
        small_sky_model = sky_model_new.create_sky_model(
            fov=config['grid']['fov'])
        sky_model = sky_model_new.full_diffuse
        energy_cfg = sky_model_new.config['grid']['energy_bin']
        sdim = config['grid']['sdim']

        def alpha(x):
            return sky_model_new.alpha_cf(x)[:sdim, :sdim]

    return small_sky_model, sky_model, alpha, energy_cfg
