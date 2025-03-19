from ...grid import Grid

from nifty8.re.library import CorrelatedMultiFrequencySky, build_default_mf_model

import numpy as np
import jax.numpy as jnp
from astropy import units as u


def _get_spectral_amplitude_model(model_cfg):
    spectral_amplitude = model_cfg.get("spectral_amplitude", {})
    return spectral_amplitude.get("model", "non_parametric")


def build_mf_model_from_grid(
    grid: Grid,
    prefix: str,
    model_cfg: dict,
    spatial_unit: u.Unit | None = u.arcsec,
    spectral_unit: u.Unit | None = u.eV,
    nonlinearity: callable = jnp.exp,
) -> CorrelatedMultiFrequencySky:

    # spatial
    shape = grid.spatial.shape
    distances = grid.spatial.distances_in(spatial_unit)

    log_energies = np.log(
        [c.to_unit(spectral_unit).value for c in grid.spectral.centers]
    )
    ref_energy = model_cfg["reference_bin"]

    spectral_amplitude_model = _get_spectral_amplitude_model(model_cfg)

    return build_default_mf_model(
        prefix=prefix,
        shape=shape,
        distances=distances,
        log_frequencies=log_energies,
        reference_frequency_index=ref_energy,
        zero_mode_settings=model_cfg.get("zero_mode"),
        spatial_amplitude_settings=model_cfg.get("spatial_amplitude"),
        spectral_index_settings=model_cfg.get("spectral_index"),
        spectral_amplitude_settings=model_cfg.get("spectral_amplitude"),
        deviations_settings=model_cfg.get("spectral_deviations"),
        spatial_amplitude_model=model_cfg["spatial_amplitude"].get(
            "model", "non_parametric"
        ),
        spectral_amplitude_model=spectral_amplitude_model,
        nonlinearity=nonlinearity,
    )
