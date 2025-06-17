import os
from dataclasses import dataclass, field

import numpy as np
import nifty8.re as jft
from radio_project_helpers.posterior_analysis import load_samples
from charm_lensing.lens_system import build_lens_system

import jubik0 as ju
from jubik0.instruments.jwst.config_handler import insert_spaces_in_lensing_new
from jubik0.instruments.jwst.parse.parsing_step import ConfigParserJwst
from jubik0.instruments.jwst.alignment.filter_alignment import FilterAlignment
from jubik0.instruments.jwst.parse.rotation_and_shift.coordinates_correction import (
    CorrectionModel,
)
from jubik0.instruments.jwst.rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
)


@dataclass
class MassAndShift:
    filter: list[str] = field(default_factory=list)
    mass_center: list[np.ndarray] = field(default_factory=list)
    shift: list[np.ndarray] = field(default_factory=list)
    shift_std: list[np.ndarray] = field(default_factory=list)

    def append_information(
        self,
        filter: str,
        mass_center: np.ndarray,
        shift: np.ndarray,
        shift_std: np.ndarray,
    ) -> None:
        self.filter.append(filter)
        self.mass_center.append(mass_center)
        self.shift.append(shift)
        self.shift_std.append(shift_std)


positive = True
base = "/home/jruestig/pro/python/j-ubik/results/jwst_lens/cluster"
filters = [
    "21_01_01_f2100w_maskcenter_HotPixels",
    "21_01_02_f1800w_maskcenter_HotPixels",
    "21_01_03_f1500w_maskcenter_HotPixels",
    "21_01_04_f1280w_maskcenter_HotPixels",
    "21_01_05_f1000w_maskcenter_HotPixels",
    "22_03_06_f770w_positiveNormal",
    "22_03_07_f560w_positiveNormal",
    "21_01_08_f444w_maskcenter_HotPixels",
    "21_01_09_f356w_maskcenter_HotPixels",
    "21_01_10_f277w_maskcenter_HotPixels",
    "21_01_11_f200w_maskcenter_HotPixels",
    "21_01_12_f150w_maskcenter_HotPixels",
    "21_01_13_f115w_maskcenter_HotPixels",
]


telescope_key = "telescope"
files_key: str = "files"

mass_and_shift = MassAndShift()
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
    center = jft.mean([convergence.parametric.prior(s)[2] for s in samples])
    shift, shift_std = jft.mean_and_std(
        [shift_and_rotation_correction.shift(s) for s in samples]
    )

    mass_and_shift.append_information(
        filter=filter_name, mass_center=center, shift=shift, shift_std=shift_std
    )


def calculate_filter_shifts(
    mass_center_reference: np.ndarray, mass_and_shift: MassAndShift, positive: bool
):
    new_shifts = dict()
    for filter, shift, shift_std, mass in zip(
        mass_and_shift.filter,
        mass_and_shift.shift,
        mass_and_shift.shift_std,
        mass_and_shift.mass_center,
    ):
        delta = mass - mass_center_reference
        if positive:
            new_shifts[filter] = dict(shift=shift + delta, shift_std=shift_std)
        elif not positive:  # negative
            new_shifts[filter] = dict(shift=shift - delta, shift_std=shift_std)

    return new_shifts


def transform_to_correction_and_shift(
    new_shifts: dict[str, np.ndarray], positive: bool
):
    if positive:
        print("#positive")
    elif not positive:  # negative
        print("#negative")

    for k, v in new_shifts.items():
        s = [[float(vvv) for vvv in vv] for vv in v["shift"]]
        v = [[float(vvv) for vvv in vv] for vv in v["shift_std"]]

        print(f"{k}:")
        print("  rotation: ['delta', 0.0, 0.1]")
        print(f"  shift: ['normal', {s}, {v}]")
        # print()


print()

transform_to_correction_and_shift(
    calculate_filter_shifts(mass_and_shift.mass_center[7], mass_and_shift, positive),
    positive=positive,
)
