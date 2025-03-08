import nifty8.re as jft


def build_single_correlated_field(
    prefix: str,
    shape: tuple[int, int],
    distances: tuple[float, float],
    zero_mode_config: dict,
    fluctuations_config: dict,
):
    cfm = jft.CorrelatedFieldMaker(f'{prefix} ')
    cfm.set_amplitude_total_offset(**zero_mode_config)
    cfm.add_fluctuations(shape, distances=distances, **fluctuations_config)
    amps = cfm.get_normalized_amplitudes()
    cfm = cfm.finalize()
    additional = {f"amplitude of {prefix}": amps}
    return cfm, additional
