import jax.numpy as jnp
from jax import random
import jubik0 as ju


if __name__ == '__main__':
    # Model settings
    shape = (256, )*2 # shape of the spatial domain
    distances = 0.1
    freqs, reference_frequency_index = jnp.array((0.1, 1.5, 2, 10)), 1

    # Random seed settings
    seed = 42
    key = random.PRNGKey(seed)

    ### PRIOR SETTINGS
    # Zero mode settings
    zero_mode_settings = (-3.0, 0.1)

    # Spatial amplitude settings
    # Option 1) Matérn correlated field
    amplitude_settings = dict(
        scale=(0.4, 0.02),
        cutoff=(0.1, 0.01),
        loglogslope=(-4, 0.1),
    )
    amplitude_model = "matern"

    # Option 2) Correlated field
    amplitude_settings = dict(
        fluctuations=(1.0, 0.02),
        loglogavgslope=(-4, 0.1),
        flexibility=None,
        asperity=None,
    )
    amplitude_model = "non_parametric"

    # Spectral amplitude settings
    # Option 1) Correlated field (could also be Matérn)
    spectral_amplitude_settings = dict(
        fluctuations=(1.0, 0.02),
        loglogavgslope=(-2., 0.1),
        flexibility=None,
        asperity=None,
    )
    spectral_amplitude_model = "non_parametric"

    # Option 2) None -> in this case it is the same as the spatial amplitude
    # spectral_amplitude_settings = None

    # Spectral index settings
    spectral_idx_settings = dict(
        mean=(-1., .05),
        fluctuations=(.1, 1.e-2),
    )

    # Deviations from power law settings
    # Option 1) Wiener process
    deviations_settings = dict(
        process='wiener',
        sigma=(0.2, 0.08),
    )
    # Option 2) None
    # deviations_settings = None

    ### BUILD MODEL
    mf_model = ju.build_default_mf_model(
        prefix='test',
        shape=shape,
        distances=distances,
        log_frequencies=freqs,
        reference_frequency_index=reference_frequency_index,
        zero_mode_settings=zero_mode_settings,
        spatial_amplitude_settings=amplitude_settings,
        spectral_index_settings=spectral_idx_settings,
        deviations_settings=deviations_settings,
        spatial_amplitude_model=amplitude_model,
        spectral_amplitude_settings=spectral_amplitude_settings,
        spectral_amplitude_model=spectral_amplitude_model
    )


    random_pos = mf_model.init(key)
    ju.plot_result(mf_model.reference_frequency_distribution(random_pos),
                   n_rows=1, n_cols=1, figsize=(15, 5),
                   title='Reference frequency distribution',)
    ju.plot_result(mf_model.spectral_index_distribution(random_pos),
                   n_rows=1, n_cols=1, figsize=(15, 5),
                   title='Spectral index distribution',)
    ju.plot_result(mf_model.spectral_deviations_distribution(random_pos),
                   n_rows=1, n_cols=freqs.shape[0], figsize=(15, 5),
                   title='Spectral deviations distribution',)
    ju.plot_result(mf_model(random_pos), n_rows=1, n_cols=freqs.shape[0],
                   figsize=(15, 5), title='MF model realization',)
