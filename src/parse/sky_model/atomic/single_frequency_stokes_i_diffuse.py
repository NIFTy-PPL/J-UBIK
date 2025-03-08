from diffuse_primatives import ZeroModelModel, FluctuationsCorrelatedFieldModel
from abstract_atomic_model import AtomicModel

from configparser import ConfigParser
from dataclasses import dataclass


@dataclass
class SingleFrequencyStokesIDiffuseModel(AtomicModel):
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

        return SingleFrequencyStokesIDiffuseModel(
            zero_mode_model=zero_mode_model,
            fluctuations_model=fluctuations_model,
        )

    @classmethod
    def from_yaml_dict(cls, prefix: str, sky_cfg: dict):
        raise NotImplementedError
