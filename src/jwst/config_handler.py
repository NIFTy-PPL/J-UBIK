from typing import Tuple

import nifty8.re as jft
import jax.numpy as jnp

from astropy.coordinates import SkyCoord
from astropy import units

from .reconstruction_grid import Grid
from ..library.sky_models import SkyModel
from ..library.sky_colormix import (
    build_colormix_components, prior_samples_colormix_components)


def _get_world_location(config: dict) -> SkyCoord:
    ra = config['grid']['pointing']['ra']
    dec = config['grid']['pointing']['dec']
    frame = config['grid']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['grid']['pointing'].get('unit', 'deg'))
    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def _get_shape(config: dict) -> Tuple[int, int]:
    npix = config['grid']['sdim']
    return (npix, npix)


def _get_fov(config: dict) -> Tuple[units.Quantity, units.Quantity]:
    fov = config['grid']['fov']
    unit = getattr(units, config['grid'].get('fov_unit', 'arcsec'))
    return fov*unit


def _get_rotation(config: dict) -> units.Quantity:
    rotation = config['grid']['pointing']['rotation']
    unit = getattr(units, config['grid']['pointing'].get('unit', 'deg'))
    return rotation*unit


def build_reconstruction_grid_from_config(config: dict) -> Grid:
    wl = _get_world_location(config)
    fov = _get_fov(config)
    shape = _get_shape(config)
    rotation = _get_rotation(config)
    return Grid(wl, shape, (fov.to(units.deg), fov.to(units.deg)), rotation=rotation)


def build_sky_model_from_config(
        config: dict, reconstruction_grid: Grid, plot=False) -> jft.Model:

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

        alpha = sky_model_new.alpha_cf

    return small_sky_model, sky_model, alpha, energy_cfg


def config_transform(config: dict):
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = eval(val)
            except:
                continue
        elif isinstance(val, dict):
            config_transform(val)


def define_mock_output(config: dict):
    from numpy import hypot
    from os.path import join

    reco_shape = config['mock_setup']['reco_shape']

    rot_string = 'r' + '_'.join([f'({r}*{rr})' for r, rr in zip(
        config['mock_setup']['rotations'],
        config['mock_setup']['reported_rotations'],
    )])
    shi_string = 's' + '_'.join([f'({hypot(*s)}*{hypot(*rs)})' for s, rs in zip(
        config['mock_setup']['shifts'],
        config['mock_setup']['reported_shifts'],
    )])

    model = config['telescope']['rotation_and_shift']['model']
    subsample = config['telescope']['rotation_and_shift']['subsample']
    method_string = model + f'{subsample}'

    output_attach = config['output']['output_attach']
    return join(
        config['output']['dir'],
        f'{reco_shape}pix',
        f'rot{rot_string}_shf{shi_string}',
        f'{method_string}_{output_attach}')
