"""
JWST DEMO
------------

This demo showcases how to reconstruct mock JWST data
using the provided configuration file.
The `jwst_demo.yaml` config file, located in `demos/configs/`,
can be modified by the user to suit their specific observation setup.

Requirements
------------
Before running this demo, ensure the following installations and downloads
are complete:

1. Install [JWST](https://jwst-pipeline.readthedocs.io/en/latest/install.html)
2. Install [webbpsf](https://webbpsf.readthedocs.io/en/stable/installation.html).
Remember to download and unzip the [webbpsf-data](https://stsci.box.com/shared/static/qxpiaxsjwo15ml6m4pkhtk36c9jgj70k.gz)
and specify the path to it in the config.
3. Install [gwcs](https://gwcs.readthedocs.io/en/latest/#installation).

YAML Configuration File Structure
---------------------------------
In the `JWST_demo.yaml` config file, you can define several parameters.
Below is a breakdown of key settings:

- **seed**: Random seed for generating reproducible results.

- **files**:
 - `res_dir`: Directory where the results of the run will be saved.
 - `webbpsf_path`: Path to the webbpsf-data folder.
 - `psf_library`: Path to the psf_library folder in which intermediate
 psf files will be stored.

- **telescope**:
  - `pointing_pixel`: Pixel position of the pointing (in pixels).
  - `fov`: Field of view (in arcsec).
  - `subsample`: Subsampling factor for the data grid

- **plotting**: Settings for output plots.
  - `enabled`: Enable or disable plotting of data.
  - `slice`: Specify the slice of data to plot (optional).
  - `dpi`: DPI setting for plot resolution.

- **grid**: Parameters defining the data grid for processing.
  - `sdim`: Spatial dimensions for the image grid.
  - `energy_bin`: Energy binning with `e_min`, `e_max`, and `e_ref` values
  for each bin.

- **priors**: Parameters defining the prior distributions for the SkyModel.
  - **point_sources**: Prior parameters for point sources in the sky model.
    - `spatial`:
      - `alpha`: The `alpha` parameter in the Inverse-Gamma distribution.
      - `q`: The `q` parameter in the Inverse-Gamma distribution.
      - `key`: A unique identifier for the point sources (e.g., "points").
    - `plaw`:
      - `mean`: Mean value for the power-law distribution of point sources.
      - `std`: Standard deviation for the power-law distribution.
      - `name`: A prefix for naming the point source power-law component.
    - `dev_wp`:
      - `x0`: Initial value for the Wiener process deviations of point sources
      along the energy axis.
      - `sigma`: A list defining the standard deviations for deviations in the
      point source model.
      - `name`: A name for the point source deviation component.

  - **diffuse**: Prior parameters for the diffuse emission component.
    - `spatial`:
      - `offset`:
        - `offset_mean`: Mean value for the offset in the spatial component of
        diffuse emission.
        - `offset_std`: Standard deviation for the offset.
      - `fluctuations`: Parameters controlling fluctuations in the diffuse
      spatial emission.
        - `fluctuations`: List of fluctuation priors (mean, std).
        - `loglogavgslope`: List of log-log average slope of fluctuations (mean, std).
        - `flexibility`: List of flexibility priors (mean, std).
        - `asperity`: Controls the roughness of the spatial power spectrum
        (set to `Null` if not used).
        - `non_parametric_kind`: Specifies the type of non-parametric model
        used for fluctuations (e.g., "power").
      - `prefix`: A prefix used for naming the diffuse spatial component.

    - `plaw`: Power-law prior for the diffuse emission component.
    Similar as for the diffuse component.

- **minimization**: Settings for the minimization algorithm used for likelihood
estimation.
  - `resume`: Whether to resume a previous minimization run.
  - `n_total_iterations`: Total number of iterations for minimization.
  - Additional parameters to control the sampling and KL-divergence
  calculations.

How to Run the Demo
-------------------
1. Ensure you have installed webbpsf correctly.
2. Modify the `jwst_demo.yaml` file to reflect your observation setup.
3. Run the demo from the command line as follows:

    ```
    python jwst_demo.py [config_file]
    ```

    If no config file is provided, the default
    `configs/jwst_demo.yaml` will be used (mock data reconstruction).

4. The script will load the configuration, process the observation data, and
   save the processed data and plots (if enabled) in the specified
   output directory.
"""
import argparse
import os
from functools import reduce
from os.path import join

import astropy.units as u
import nifty8.re as jft
import numpy as np
from astropy.coordinates import SkyCoord
from jax import config, random

import jubik0 as ju
from jubik0.instruments.jwst.filter_projector import FilterProjector
from jubik0.likelihood import (build_gaussian_likelihood,
                               connect_likelihood_to_model)

config.update('jax_enable_x64', True)


# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help="Config file (.yaml) for JWST inference.",
                    nargs='?', const=1, default="configs/jwst_demo.yaml")
args = parser.parse_args()

filters = dict(
    f200w=dict(distance=0.031, key=0),
    f150w=dict(distance=0.031, key=1),
    f115w=dict(distance=0.031, key=2),
)
pointing_center = (0, 0)

if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    # Uncomment to save local packages git hashes to file
    # ju.save_local_packages_hashes_to_txt(
    #     ['jubik0', 'nifty8'],
    #     os.path.join(file_info['res_dir'], "packages_hashes.txt"),
    #     paths_to_git=[os.path.dirname(os.getcwd()), None],
    #     verbose=False)

    # Save run configuration
    ju.copy_config(os.path.basename(config_path),
                   path_to_yaml_file=os.path.dirname(config_path),
                   output_dir=file_info['res_dir'])

    # Load sky model
    sky_model = ju.SkyModel(cfg)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # Draw random numbers
    key = random.PRNGKey(cfg['seed'])

    energy_cfg = cfg['grid']['energy_bin']
    e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
    filter_projector = FilterProjector(
        sky_domain=sky.target,
        keys_and_colors={key: val['key'] for key, val in filters.items()},
    )

    key, subkey = random.split(key)
    sky_model_with_filters = jft.Model(
        lambda x: filter_projector(sky(x)),
        domain=sky.domain)

    mock_sky = sky_model_with_filters(sky_model_with_filters.init(subkey))
    reconstruction_grid = ju.Grid(
        center=SkyCoord(pointing_center[0]*u.rad, pointing_center[1]*u.rad),
        shape=(cfg['grid']['sdim'],)*2,
        fov=(cfg['grid']['fov']*u.arcsec,)*2
    )

    all_filter_data = []
    likelihoods = []
    for fltname in filters.keys():

        psf_kwargs = dict(
            camera='nircam',
            filter=fltname,
            center_pixel=cfg['telescope']['pointing_pixel'],
            webbpsf_path=file_info['webbpsf_path'],
            psf_library_path=file_info['psf_library'],
            fov_pixels=cfg['telescope']['fov'],
        )

        data_fov = cfg['telescope']['fov']
        data_shape = int(data_fov/filters[fltname]['distance'])
        data_grid = ju.Grid(
            center=SkyCoord(
                pointing_center[0]*u.rad, pointing_center[1]*u.rad),
            shape=(data_shape,)*2,
            fov=(data_fov*u.arcsec,)*2
        )
        rotation_and_shift_kwargs = dict(
            reconstruction_grid=reconstruction_grid,
            data_dvol=data_grid.dvol,
            data_wcs=data_grid.wcs,
            data_model_type='linear',
            world_extrema=data_grid.world_extrema(),
        )

        response = ju.build_jwst_response(
            sky_domain={fltname: filter_projector.target[fltname]},
            subsample=cfg['telescope']['subsample'],
            rotation_and_shift_kwargs=rotation_and_shift_kwargs,
            psf_kwargs=psf_kwargs,
            data_mask=None,
            transmission=1.,
            zero_flux=None,
        )

        # Create mock data
        noise_std = cfg['mock_config']['noise_std']
        key, subkey = random.split(key)
        data = (
            response(mock_sky) +
            random.normal(subkey, response.target.shape) * noise_std
        )
        all_filter_data.append(data)

        likelihood = build_gaussian_likelihood(data, noise_std)
        likelihood = likelihood.amend(
            response, domain=jft.Vector(response.domain))
        likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x+y, likelihoods)
    likelihood = connect_likelihood_to_model(
        likelihood, sky_model_with_filters)

    all_filter_data = np.array(all_filter_data)
    ju.plot_result(all_filter_data, output_file=join(file_info["res_dir"],
                                                     "data.png"))

    # Plot
    additional_plot_dict = {}
    if hasattr(sky_model, 'alpha_cf'):
        additional_plot_dict['diffuse_alfa'] = sky_model.alpha_cf
    if hasattr(sky_model, 'points_alfa'):
        additional_plot_dict['points_alfa'] = sky_model.points_alfa

    def simple_eval_plots(s, x):
        """Call plot_sample_and_stat for every iteration."""
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 dpi=cfg["plotting"]["dpi"],
                                 iteration=x.nit,
                                 rgb_min_sat=[3e-8, 3e-8, 3e-8],
                                 rgb_max_sat=[2.0167e-6, 1.05618e-6, 1.5646e-6])
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 additional_plot_dict,
                                 s,
                                 dpi=cfg["plotting"]["dpi"],
                                 iteration=x.nit,
                                 log_scale=False,
                                 plot_samples=False,
                                 )
        ju.plot_pspec(sky_model.spatial_pspec,
                      sky_model.spatial_cf.target.shape,
                      sky_model.s_distances,
                      s,
                      file_info["res_dir"],
                      iteration=x.nit,
                      dpi=cfg["plotting"]["dpi"],
                      )

    # Minimization
    minimization_config = cfg['minimization']
    n_dof = ju.get_n_constrained_dof(likelihood)
    minimization_parser = ju.MinimizationParser(minimization_config,
                                                n_dof=n_dof)
    key, subkey = random.split(key)
    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    samples, state = jft.optimize_kl(
        likelihood,
        pos_init,
        key=key,
        n_total_iterations=minimization_config['n_total_iterations'],
        resume=minimization_config['resume'],
        n_samples=minimization_parser.n_samples,
        draw_linear_kwargs=minimization_parser.draw_linear_kwargs,
        nonlinearly_update_kwargs=minimization_parser.nonlinearly_update_kwargs,
        kl_kwargs=minimization_parser.kl_kwargs,
        sample_mode=minimization_parser.sample_mode,
        callback=simple_eval_plots,
        odir=file_info["res_dir"], )
