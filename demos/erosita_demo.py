"""
eROSITA DEMO
------------

This demo showcases how to process eROSITA data using eSASS (eROSITA Science
Analysis Software System) and the provided configuration file.
The `eROSITA_demo.yaml` config file, located in `demos/configs/`,
can be modified by the user to suit their specific observation setup.

Requirements
------------
Before running this demo, ensure the following installations and downloads
are complete:

1. Download [eSASS](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/)
Docker image and ensure that the image is running.
2. Download the eROSITA CALDB (Calibration Database) folder:
   [CALDB](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/caldb4DR1.tgz)
3. Download eROSITA data:
   - Early Data Release (EDR): [eROSITA EDR](https://erosita.mpe.mpg.de/edr/index.php)
   - Data Release 1 (DR1): [eROSITA DR1](https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/)

YAML Configuration File Structure
---------------------------------
In the `eROSITA_demo.yaml` config file, you need to define several parameters
based on your observation data and settings.
Below is a breakdown of key settings:

- **seed**: Random seed for generating reproducible results.

- **esass_image**: Version of the eSASS image (DR1 or EDR) you have installed.

- **files**:
  - `obs_path`: Path to the folder containing the downloaded eROSITA data
   (e.g., '../data/LMC_SN1987A/').
  - `data_dict`: Name of the pickle file storing the processed observation data
  (e.g., 'data.pkl').
  - `processed_obs_folder`: Directory where processed observation files are
  stored (e.g., 'processed'). Only needed for real data.
  - `input`: Name of the input event list file
  (e.g., 'pm00_700161_020_EventList_c001.fits').
  - `output`: Name of the processed data output file
  (e.g., 'pm00_700161_020_data.fits').
  - `exposure`: Name of the exposure map file created after processing
  (e.g., 'pm00_700161_020_expmap.fits').
  - `calibration_path`: Path to the folder where the CALDB calibration data is
  located (e.g., '../data/').
  - `caldb_folder_name`: Name of the CALDB folder (e.g., 'caldb').
  - `psf_filename_suffix`: Suffix for PSF (Point Spread Function) files.
  - `effective_area_filename_suffix`: Suffix for effective area files.
  - `res_dir`: Directory where the results of the run will be saved
  (e.g., 'results/my_results'). Since the final processed data will be
  saved here, a new res_dir name should be chosen if the minimization parameter
  `resume` is set to `False`.
  - `mock_gen_config`: Path to the mock data generation configuration file
  (e.g., 'configs/erosita_demo.yaml'). Only needed for simulated data.
  Can be a different file than the reconstruction config.
  - `pos_dict`: Name of the file storing position data (e.g., 'pos.pkl').
  Only needed for simulated data.

- **telescope**:
  - `tm_ids`: List of Telescope Module (TM) IDs to use (e.g., [1, 2, 3, 4]).
  - `rebin`: Rebin parameter for data processing (see eSASS evtool
  documentation).
  - `pattern`: Pattern parameter for data processing (see eSASS
  evtool documentation).
  - `detmap`: Flag to enable or disable the use of detector maps.
  - `exp_cut`: Lower exposure threshold in seconds.
  - `badpix_correction: Whether to correct the exposure maps for badpixels
  that might not be accounted for by eSASS. If true, you will need to produce
        new detmaps. See `create_erosita_badpix_to_detmaps` for details.
  - `effective_area_correction`: Whether to correct the exposure maps for
  the detector's effective area.

- **psf**:
  - `energy`: List of energy levels (in keV) at which PSF files are defined
  (e.g., ['0277', '1486', '3000']).
  - Additional PSF method-specific settings, such as `npatch`, `margfrac`, etc.

- **plotting**: Settings for output plots.
  - `priors`: Enable or disable plotting of prior distributions.
  - `priors_signal_response`: Enable or disable plotting of prior signal
  response.
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
1. Ensure you have installed eSASS in a Docker container and downloaded the
required data.
2. Modify the `eROSITA_demo.yaml` file to reflect your observation setup.
3. Run the demo from the command line as follows:

    ```
    python eROSITA_demo.py [config_file]
    ```

    If no config file is provided, the default
    `configs/eROSITA_demo.yaml` will be used (mock data reconstruction).

4. The script will load the configuration, process the observation data, and
   save the processed data and plots (if enabled) in the specified
   output directory.

Notes
-----
- If processing simulated data, specify the `mock_gen_config` and `pos_dict` in
the configuration file.
- If during a new reconstruction an old results directory is used, the
"data.pkl" file will be sourced. If any telescope- or data-related
parameters are changed, the "data.pkl" file must be deleted or the results
directory renamed.
- If resuming a previous run, the output directory will be checked,
and a new directory will be created
  if one already exists.

References
----------
- [eROSITA eSASS Installation Guide](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/)
- [eROSITA Data Release 1 (DR1)](https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/)
"""

import argparse
import os
from os.path import join

import nifty8.re as jft
from jax import config, random

import jubik0 as ju

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config',
                    type=str,
                    help="Config file (.yaml) for eROSITA inference.",
                    nargs='?',
                    const=1,
                    default="configs/eROSITA_demo.yaml")
args = parser.parse_args()
if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']
    plot_info = cfg['plotting']

    # Save run configuration
    ju.copy_config(os.path.basename(config_path),
                   path_to_yaml_file=os.path.dirname(config_path),
                   output_dir=file_info['res_dir'])

    # Uncomment to save local packages git hashes to file
    # ju.save_local_packages_hashes_to_txt(
    #     ['jubik0', 'nifty8'],
    #     join(file_info['res_dir'], "packages_hashes.txt"),
    #     paths_to_git=[os.path.dirname(os.getcwd()), None],
    #     verbose=False)

    # Load sky model
    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # Generate eROSITA data (if it does not already exist)
    ju.generate_erosita_data_from_config(config_path)

    # TODO shift to ju.generate_erosita_likelihood.amend(sky)
    # Generate loglikelihood (Building masked (mock) data and response)
    log_likelihood = ju.generate_erosita_likelihood_from_config(config_path,
                                                                sky)

    # Set initial position
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    # Plot priors
    if plot_info['priors']:
        if 'prior_plot_dir' in file_info:
            prior_plot_dir = join(file_info['res_dir'],
                                  file_info['prior_plot_dir'])
        else:
            raise ValueError(
                "The 'prior_plot_dir' parameter must be specified in "
                "the 'files' section of the config file.")
        ju.plot_erosita_priors(subkey,
                               plot_info['n_prior_samples'],
                               config_path,
                               prior_plot_dir,
                               plot_info['priors_signal_response'],
                               adjust_figsize=True,
                               )

    # Minimization
    minimization_config = cfg['minimization']
    n_dof = ju.get_n_constrained_dof(log_likelihood)
    minimization_parser = ju.MinimizationParser(minimization_config,
                                                n_dof=n_dof)

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


    samples, state = jft.optimize_kl(
        log_likelihood,
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
