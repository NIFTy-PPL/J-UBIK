import yaml
import os
import pickle
from os.path import join

from jax import random

import nifty8.re as jft
import jubik0 as ju
from jubik0.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    insert_spaces_in_lensing)
from charm_lensing.lens_system import build_lens_system


def get_pretrain_data(model_cfg: dict, full_mass=False):
    '''Returns the
        - lens_mass (mean, std)
        - lens_light (mean, std)
        - source_light (mean, std)
    for the last iteration and model of the pretrain_path of the model_cfg.'''

    pretrain_cfg_path = model_cfg['files']['pretrain_path']
    cfg = yaml.load(open(pretrain_cfg_path, 'r'), Loader=yaml.SafeLoader)

    insert_spaces_in_lensing(cfg)
    lens_system = build_lens_system(cfg['lensing'])

    odir = cfg['files']['res_dir']
    LAST_FILENAME = "last.pkl"
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None

    with open(last_fn, "rb") as f:
        samples, opt_vi_st = pickle.load(f)

    mass = lens_system.lens_plane_model.convergence_model if full_mass else lens_system.lens_plane_model.convergence_model.parametric()
    lens_mass = jft.mean_and_std([mass(s) for s in samples])

    lens_light = lens_system.lens_plane_model.light_model
    lens_light = jft.mean_and_std([lens_light(s) for s in samples])

    source_light = lens_system.source_plane_model.light_model
    source_light = jft.mean_and_std([source_light(s) for s in samples])

    return lens_mass, lens_light, source_light


def pretrain_model(cfg, lens_system, plot):

    pretrain_res_dir = join(cfg['files']['res_dir'], 'pretrain')
    lens_mass, lens_light, source_light = get_pretrain_data(cfg)

    lens_mass_model = lens_system.lens_plane_model.convergence_model
    lens_light_model = lens_system.lens_plane_model.light_model
    source_light_model = lens_system.source_plane_model.light_model

    likelihood_lens_mass = ju.library.likelihood.build_gaussian_likelihood(
        lens_mass[0], lens_mass[1])
    likelihood_lens_mass = likelihood_lens_mass.amend(
        lens_mass_model, domain=jft.Vector(lens_mass_model.domain))

    likelihood_lens_light = ju.library.likelihood.build_gaussian_likelihood(
        lens_mass[0], lens_mass[1])
    likelihood_lens_light = likelihood_lens_light.amend(
        lens_light_model, domain=jft.Vector(lens_light_model.domain))

    likelihood_source = ju.library.likelihood.build_gaussian_likelihood(
        source_light[0], source_light[1])
    likelihood_source = likelihood_source.amend(
        source_light_model, domain=jft.Vector(source_light_model.domain))

    likelihood = likelihood_source + likelihood_lens_mass + likelihood_lens_light

    cfg_mini = ju.get_config(config_path)["minimization"]
    n_dof = ju.calculate_n_constrained_dof(likelihood)
    minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
    key = random.PRNGKey(cfg_mini.get('key', 42))
    key, rec_key = random.split(key, 2)
    pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

    print(f'Results: {pretrain_res_dir}')
    samples, state = jft.optimize_kl(
        likelihood,
        pos_init,
        key=rec_key,
        callback=plot,
        odir=pretrain_res_dir,
        n_total_iterations=cfg_mini['pretraining_steps'],
        n_samples=minpars.n_samples,
        sample_mode=minpars.sample_mode,
        draw_linear_kwargs=minpars.draw_linear_kwargs,
        nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
        kl_kwargs=minpars.kl_kwargs,
        resume=cfg_mini.get('resume', False),
    )
    return samples


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Config File",
        type=str,
        nargs='?',
        const=1,
        default='./demos/jwst_lens_config.yaml')
    args = parser.parse_args()
    config_path = args.config

    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
    res_dir = cfg['files']['res_dir']
    os.makedirs(res_dir, exist_ok=True)
    ju.save_local_packages_hashes_to_txt(
        ['nifty8', 'charm_lensing', 'jubik0'],
        os.path.join(res_dir, 'hashes.txt'))
    insert_spaces_in_lensing(cfg)
    lens_system = build_lens_system(cfg['lensing'])

    samples = pretrain_model(cfg, lens_system, plot=None)
