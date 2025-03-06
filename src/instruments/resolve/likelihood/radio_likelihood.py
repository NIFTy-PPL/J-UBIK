from ...grid import Grid

from astropy import units as u


def build_radio_likelihood(
    data_names: list[str],
    cfg: dict,
    sky_grid: Grid,
    sky_model: jft.Model,
    last_radio_bin: int | None,
    sky_key: str = 'sky',
    sky_unit: u.Unit | None = None,
):
    radio_sky_extractor_resolved, radio_grid = build_radio_sky_extractor(
        last_radio_bin,
        sky_model,
        sky_grid,
        sky_key=sky_key,
        sky_unit=sky_unit)

    response_backend_settings = yaml_to_response_settings(
        cfg['radio_response'])

    if not isinstance(data_names, list):
        assert isinstance(data_names, str)
        data_names = [data_names]

    likelihoods = []
    sky_beamers = []
    for data_name in data_names:
        logger.info(f'Loading data: {data_name}')
        observations = list(load_and_modify_data_from_objects(
            sky_frequencies=radio_grid.spectral.binbounds_in(u.Hz),
            data_loading=yaml_to_data_loading(cfg['alma_data'][data_name]),
            observation_modify=yaml_to_observation_modify(
                cfg['alma_data'][data_name])
        ))

        likelihood, _sky_beamer = build_jax_instrument_likelihood(
            yaml_to_beam_pattern(cfg['alma_data'][data_name]),
            data_kw=data_name,
            sky_shape_with_dtype=radio_sky_extractor_resolved.target,
            sky_grid=radio_grid,
            observations=observations,
            backend_settings=response_backend_settings,
            direction_key='PHASE_DIR',
        )
        likelihoods.append(likelihood)
        sky_beamers.append(_sky_beamer)
        logger.info('')

    likelihood = reduce(lambda x, y: x+y, likelihoods)
    _sky_beamer = reduce(lambda x, y: x+y, sky_beamers)

    # Wrapped call
    sky_beamer = jft.Model(
        lambda x: _sky_beamer(radio_sky_extractor_resolved(x)),
        domain=radio_sky_extractor_resolved.domain)
    return likelihood, sky_beamer, radio_sky_extractor_resolved
