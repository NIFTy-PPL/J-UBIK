from jubik.parse.sky_model.multifrequency.spectral_product_mf_sky import (
    SimpleSpectralSkyConfig,
)


def test_null_spectral_amplitude():
    cfg = SimpleSpectralSkyConfig.from_yaml_dict(
        dict(
            reference_bin=0,
            zero_mode=(0.0, 1.0),
            spatial_amplitude={},
            spectral_index={},
            spectral_amplitude=None,
        )
    )
    assert cfg.spectral_amplitude is None
    assert cfg.spectral_amplitude_model == "non_parametric"
