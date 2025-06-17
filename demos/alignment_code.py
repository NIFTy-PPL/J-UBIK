import os
from sys import exit
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import nifty8.re as jft
from charm_lensing.lens_system import build_lens_system
from jax import config, devices
from radio_project_helpers.posterior_analysis import load_samples
from radio_project_helpers.results import save_as_fits
from charm_lensing.utils import save_fits
from astropy import units as u

import jubik0 as ju
from jubik0.instruments.jwst.config_handler import insert_spaces_in_lensing_new
from jubik0.parse.grid import GridModel
from jubik0.sky_model.multifrequency.mf_model_from_grid import build_mf_model_from_grid
from jubik0.instruments.jwst.parse.parsing_step import ConfigParserJwst
from jubik0.instruments.jwst.alignment.filter_alignment import FilterAlignment
from jubik0.instruments.jwst.parse.rotation_and_shift.coordinates_correction import (
    CorrectionModel,
)
from jubik0.instruments.jwst.rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
)

config.update("jax_default_device", devices("cpu")[0])

SKY_KEY = "sky"

minimization_config = {
    "key": 20250602,
    "resume": True,
    "n_total_iterations": 5,
    "delta": {
        "switches": [0],
        "values": [5.0e-9],
    },
    "samples": {
        "switches": [0],
        "n_samples": [0],
        "mode": ["linear_resample"],
        "lin_maxiter": [250],
        "nonlin_maxiter": [7],
    },
    "kl_minimization": {
        "switches": [0],
        "kl_xtol": 1.0e-12,
        "kl_maxiter": [15],
    },
}


# sky_grid = ju.Grid.from_grid_model(GridModel.from_yaml_dict(cfg_yaml["sky"]["grid"]))
# method = "lens"
# print("Method:", method)
# _sky_lens = lens_system.get_forward_model_parametric(only_source=True)
# sky_lens = jft.Model(
#     jft.wrap_left(lambda x: _sky_lens(x)[None], SKY_KEY), domain=_sky_lens.domain
# )
# lens_light = jft.mean([sky_lens(s)["sky"] for s in samples])


base = "/home/jruestig/pro/python/j-ubik/results/jwst_lens/cluster"
# filters = [
#     "18_04_f115w_fixedStars",
#     "18_04_f150w_fixedStars",
#     "18_04_f200w_fixedStars",
#     "18_04_f277w_fixedStars",
#     "18_04_f356w_fixedStars",
#     "18_02_f444w_fixedStars",
#     "18_04_f560w_fixedStars",
#     "18_04_f770w_fixedStars",
#     "18_04_f1000w_fixedStars",
#     "18_04_f1280w_fixedStars_02_highres",
#     # "18_04_f1500w_fixedStars",
#     "18_03_f1800w_fixedStars",
#     "18_03_f2100w_fixedStars",
# ]

filters = [
    "21_01_01_f2100w_maskcenter_HotPixels",
    "21_01_02_f1800w_maskcenter_HotPixels",
    "21_01_03_f1500w_maskcenter_HotPixels",
    "21_01_04_f1280w_maskcenter_HotPixels",
    "21_01_05_f1000w_maskcenter_HotPixels",
    "21_01_06_f770w_maskcenter_HotPixels",
    "21_01_07_f560w_maskcenter_HotPixels",
    "21_01_08_f444w_maskcenter_HotPixels",
    "21_01_09_f356w_maskcenter_HotPixels",
    "21_01_10_f277w_maskcenter_HotPixels",
    "21_01_11_f200w_maskcenter_HotPixels",
    "21_01_12_f150w_maskcenter_HotPixels",
    "21_01_13_f115w_maskcenter_HotPixels",
]
masses = {}


@dataclass
class PlottingInfo:
    extent: tuple[float]


@dataclass
class FilterMassShift:
    mass_field: np.ndarray
    center: np.ndarray
    shifts: np.ndarray
    shift_model: ShiftAndRotationCorrection
    plotting: PlottingInfo

    def __post_init__(self):
        self.mass_field = np.array(self.mass_field)
        self.center = np.array(self.center)
        self.shifts = np.array(self.shifts)


telescope_key = "telescope"
files_key: str = "files"
for ff in filters:
    path = os.path.join(base, ff)
    print("Loading:", path)

    # Lensing system
    config_path = os.path.join(path, "config.yaml")
    cfg_yaml = ju.get_config(config_path)
    insert_spaces_in_lensing_new(cfg_yaml["sky"])
    lens_system = build_lens_system(cfg_yaml["sky"])
    convergence = lens_system.lens_plane_model.convergence_model.parametric

    # Shift correction
    cfg_parser = ConfigParserJwst.from_yaml_dict(
        cfg_yaml, telescope_key=telescope_key, files_key=files_key
    )
    assert len(cfg_parser.data_loader.paths.keys()) == 1
    filter_name, file_paths = next(iter(cfg_parser.data_loader.paths.items()))
    filter_alignment = FilterAlignment(filter_name=filter_name)
    filter_alignment.load_correction_prior(
        cfg_yaml[telescope_key]["rotation_and_shift"]["correction_priors"],
        number_of_observations=len(file_paths),
    )
    assert filter_alignment.correction_prior.model == CorrectionModel.SHIFT
    shift_and_rotation_correction = ShiftAndRotationCorrection(
        domain_key=filter_name,
        correction_prior=filter_alignment.correction_prior,
        rotation_center=None,
    )

    # Samples
    samples = load_samples(path)

    # Get mass
    mass = jft.mean([convergence(s) for s in samples])
    center = jft.mean([convergence.parametric.prior(s)[2] for s in samples])
    shifts = jft.mean([shift_and_rotation_correction.shift(s) for s in samples])

    # save_name = f"{ff.split('_')[2]}_{ff.split('_')[1]}"
    save_name = f"{ff.split('_')[3]}"
    masses[save_name] = FilterMassShift(
        mass_field=mass,
        center=center,
        shifts=shifts,
        shift_model=shift_and_rotation_correction,
        plotting=PlottingInfo(
            extent=lens_system.lens_plane_model.space.extend().extent,
        ),
    )


fig, axes = plt.subplots(12 // 3, 12 // 4, sharex=True, sharey=True)
# fig, axes = plt.subplots(2, 3)
axes = axes.flatten()
for ax, f in zip(axes, masses.keys()):
    ax.imshow(
        masses[f].mass_field, vmax=1.5, origin="lower", extent=masses[f].plotting.extent
    )
    ax.scatter(*masses[f].center)
    ax.scatter(*(masses[f].shifts.T), marker="x", c="r")
    cc = masses[f].center
    ax.set_title(f"{f}_[{cc[0]:.3f}, {cc[1]:.3f}]")
plt.show()


if False:
    mass_centers = np.array([masses[f].center for f in masses.keys()])
    mass_center_model = jft.VModel(
        jft.Model(
            lambda x: convergence.parametric.prior(x)[2],
            domain=convergence.parametric.prior.domain,
        ),
        mass_centers.shape[0],
    )

    d = mass_centers - mass_centers[0]

    shifts_data = {f: masses[f].shifts - mass_centers[0] for f in masses.keys()}

    def fit_model(
        data,
        model,
        std=1e-8,
        cfg_mini=minimization_config,
        res_dir="tmp_output/massfitting/",
    ):
        from jax import random

        likelihood = ju.likelihood.build_gaussian_likelihood(
            data=data, std=std, model=model
        )

        n_dof = ju.get_n_constrained_dof(likelihood)

        minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
        key, rec_key = random.split(random.PRNGKey(42), 2)
        pos_init = 0.1 * jft.random_like(rec_key, likelihood.domain)

        samples, state = jft.optimize_kl(
            likelihood,
            pos_init,
            odir=res_dir,
            key=rec_key,
            n_total_iterations=2,
            n_samples=minpars.n_samples,
            sample_mode=minpars.sample_mode,
            draw_linear_kwargs=minpars.draw_linear_kwargs,
            nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
            kl_kwargs=minpars.kl_kwargs,
            resume=False,  # cfg_mini.get('resume', False),
        )

        return samples.pos

    pos = fit_model(d, mass_center_model)
