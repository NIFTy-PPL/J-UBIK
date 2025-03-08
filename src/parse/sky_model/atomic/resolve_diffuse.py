from dataclasses import dataclass
from configparser import ConfigParser


from .single_frequency_stokes_i_diffuse import SingleFrequencyStokesIDiffuseModel


@dataclass
class ResolveDiffuseSkyModel:
    prefix: str
    diffuse_config: SingleFrequencyStokesIDiffuseModel  # TODO : More options

    @classmethod
    def from_config_parser(cls, sky_cfg: ConfigParser):
        # TODO: Add more Options
        if not sky_cfg["freq mode"] == "single":
            raise NotImplementedError(
                "FIXME: only implemented for single frequency")
        if not sky_cfg["polarization"] == "I":
            raise NotImplementedError("FIXME: only implemented for stokes I")

        prefix = "stokesI diffuse space i0"
        diffuse_config = SingleFrequencyStokesIDiffuseModel.from_config_parser(
            prefix=prefix, sky_cfg=sky_cfg)

        return ResolveDiffuseSkyModel(prefix, diffuse_config)

    @classmethod
    def from_yaml_dict(cls, sky_cfg: ConfigParser):
        raise NotImplementedError
