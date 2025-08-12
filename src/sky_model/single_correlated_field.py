import nifty.re as jft
from ..parse.sky_model.correlated_field import (
    CfmFluctuationsConfig,
    MaternFluctationsConfig,
)


def build_single_correlated_field(
    prefix: str,
    shape: tuple[int, int],
    distances: tuple[float, float],
    zero_mode_config: dict,
    fluctuations_config: dict,
):
    cfm = jft.CorrelatedFieldMaker(f"{prefix} ")
    cfm.set_amplitude_total_offset(**zero_mode_config)
    cfm.add_fluctuations(shape, distances=distances, **fluctuations_config)
    amps = cfm.get_normalized_amplitudes()
    cfm = cfm.finalize()
    additional = {f"amplitude of {prefix}": amps}
    return cfm, additional


def build_single_correlated_field_from_config(
    prefix: str,
    shape: tuple[int, int],
    distances: tuple[float, float],
    config: CfmFluctuationsConfig | MaternFluctationsConfig,
) -> jft.Model:
    cfm = jft.CorrelatedFieldMaker(f"{prefix} ")
    cfm.set_amplitude_total_offset(
        offset_mean=config.amplitude.offset_mean,
        offset_std=config.amplitude.offset_std,
    )

    if isinstance(config, CfmFluctuationsConfig):
        cfm.add_fluctuations(
            shape=shape,
            distances=distances,
            fluctuations=config.fluctuations,
            loglogavgslope=config.loglogavgslope,
            flexibility=config.flexibility,
            asperity=config.asperity,
            prefix=config.prefix,
            harmonic_type=config.harmonic_type,
            non_parametric_kind=config.non_parametric_kind,
        )
    elif isinstance(config, MaternFluctationsConfig):
        cfm.add_fluctuations_matern(
            shape=shape,
            distances=distances,
            scale=config.scale,
            cutoff=config.cutoff,
            loglogslope=config.loglogslope,
            renormalize_amplitude=config.renormalize_amplitude,
            prefix=config.prefix,
            harmonic_type=config.harmonic_type,
            non_parametric_kind=config.non_parametric_kind,
        )
    else:
        raise NotImplementedError(
            f"{type(config)} unknown. Supported models: "
            "`MaternFluctationsConfig` or `CfmFluctuationsConfig`."
        )

    return cfm.finalize()
