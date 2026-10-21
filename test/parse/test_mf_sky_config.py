# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

import pytest

from jubik.parse.sky_model.multifrequency.spectral_product_mf_sky import (
    SimpleSpectralSkyConfig,
)

BASE = dict(
    reference_bin=0,
    zero_mode=(0.0, 1.0),
    spatial_amplitude={},
    spectral_index={},
    spectral_amplitude=None,
)


def test_null_spectral_amplitude():
    cfg = SimpleSpectralSkyConfig.from_yaml_dict(BASE)
    assert cfg.spectral_amplitude is None
    assert cfg.spectral_amplitude_model == "non_parametric"


def test_unsupported_harmonic_type_raises():
    with pytest.raises(ValueError):
        SimpleSpectralSkyConfig.from_yaml_dict({**BASE, "harmonic_type": "spherical"})
