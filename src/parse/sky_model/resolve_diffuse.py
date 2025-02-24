from dataclasses import dataclass
from configparser import ConfigParser


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


@dataclass
class ResolveSingleFrequencyStokesIDiffuseModel:
    zero_mode_model: ZeroModelModel
    fluctuations_model: FluctuationsCorrelatedFieldModel

    @classmethod
    def from_config_parser(cls, prefix: str, sky_cfg: ConfigParser):
        '''Parse cfg to ResolvePointSourcesModel.

        Parameters
        ----------
        prefix: str
            The prefix of the sky diffuse model.
        sky_cfg: ConfigParser
            The config file containging:
            - {prefix} zero mode offset
            - {prefix} zero mode mean
            - {prefix} zero mode stddev

            - {prefix} fluctuations mean
            - {prefix} fluctuations stddev
            - {prefix} loglogavgslope mean
            - {prefix} loglogavgslope stddev
            - {prefix} flexibility mean
            - {prefix} flexibility stddev
            - {prefix} asperity mean
            - {prefix} asperity stddev
        '''

        zmo = sky_cfg.getfloat(f"{prefix} zero mode offset")
        zmm = sky_cfg.getfloat(f"{prefix} zero mode mean")
        zms = sky_cfg.getfloat(f"{prefix} zero mode stddev")
        zero_mode_model = ZeroModelModel(
            offset_mean=zmo, offset_std=(zmm, zms)
        )

        flum = sky_cfg.getfloat(f"{prefix} fluctuations mean")
        flus = sky_cfg.getfloat(f"{prefix} fluctuations stddev")
        llam = sky_cfg.getfloat(f"{prefix} loglogavgslope mean")
        llas = sky_cfg.getfloat(f"{prefix} loglogavgslope stddev")
        flem = sky_cfg.getfloat(f"{prefix} flexibility mean")
        fles = sky_cfg.getfloat(f"{prefix} flexibility stddev")
        aspm = sky_cfg.getfloat(f"{prefix} asperity mean")
        asps = sky_cfg.getfloat(f"{prefix} asperity stddev")
        fluctuations_model = FluctuationsCorrelatedFieldModel(
            fluctuations=(flum, flus),
            loglogavgslope=(llam, llas),
            flexibility=(flem, fles),
            asperity=(aspm, asps),
        )

        return ResolveSingleFrequencyStokesIDiffuseModel(
            zero_mode_model=zero_mode_model,
            fluctuations_model=fluctuations_model,
        )

    @classmethod
    def from_yaml_dict(cls, prefix: str, sky_cfg: dict):
        raise NotImplementedError


@dataclass
class ResolveDiffuseSkyModel:
    prefix: str
    diffuse_config: ResolveSingleFrequencyStokesIDiffuseModel

    @classmethod
    def from_config_parser(cls, sky_cfg: ConfigParser):
        # TODO: Add more Options
        if not sky_cfg["freq mode"] == "single":
            raise NotImplementedError(
                "FIXME: only implemented for single frequency")
        if not sky_cfg["polarization"] == "I":
            raise NotImplementedError("FIXME: only implemented for stokes I")

        prefix = "stokesI diffuse space i0"
        diffuse_config = ResolveSingleFrequencyStokesIDiffuseModel.from_config_parser(
            prefix=prefix, sky_cfg=sky_cfg)

        return ResolveDiffuseSkyModel(prefix, diffuse_config)

    @classmethod
    def from_yaml_dict(cls, sky_cfg: ConfigParser):
        raise NotImplementedError
