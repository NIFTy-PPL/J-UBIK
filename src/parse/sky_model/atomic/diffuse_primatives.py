from dataclasses import dataclass


@dataclass
class ZeroModelModel:
    offset_mean: float
    offset_std: tuple[float, float]


@dataclass
class FluctuationsCorrelatedFieldModel:
    fluctuations: tuple[float, float]
    loglogavgslope: tuple[float, float]
    flexibility: tuple[float, float]
    asperity: tuple[float, float]
    harmonic_type: str = 'Fourier'
    prefix: str = ''
    non_parametric_kind: str = 'power'
