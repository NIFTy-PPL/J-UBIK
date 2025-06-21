from dataclasses import dataclass

from .zero_flux_model import ZeroFluxPriorConfigs
from .rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
    LinearConfig,
    NufftConfig,
)
from .variable_covariance import (
    VariableCovarianceConfig,
    variable_covariance_config_factory,
)
from .alignment.star_alignment import StarAlignmentConfig
from .data.data_loader import DataLoadingConfig, Subsample
from .jwst_psf import JwstPsfKernelConfig
from .masking.data_mask import ExtraMasks


@dataclass
class ConfigParserJwst:
    zero_flux_prior_configs: ZeroFluxPriorConfigs
    rotation_and_shift_algorithm: LinearConfig | NufftConfig
    psf_kernel_configs: JwstPsfKernelConfig
    data_loader: DataLoadingConfig
    subsample_target: Subsample
    extra_masks: ExtraMasks
    star_alignment_config: StarAlignmentConfig | None
    variable_covariance_config: VariableCovarianceConfig | None

    @classmethod
    def from_yaml_dict(
        cls, cfg: dict, telescope_key="telescope", files_key="files"
    ) -> "ConfigParserJwst":
        """Parse the jwst config."""

        zero_flux_prior_configs = ZeroFluxPriorConfigs.from_yaml_dict(
            cfg[telescope_key].get("zero_flux")
        )
        rotation_and_shift_algorithm = rotation_and_shift_algorithm_config_factory(
            cfg[telescope_key]["rotation_and_shift"]
        )
        psf_kernel_configs = JwstPsfKernelConfig.from_yaml_dict(
            cfg[telescope_key].get("psf")
        )
        data_loader = DataLoadingConfig.from_yaml_dict(cfg[files_key])
        extra_masks = ExtraMasks.from_yaml_dict(cfg[telescope_key])

        star_alignment_config: StarAlignmentConfig | None = (
            StarAlignmentConfig.from_yaml_dict(cfg[telescope_key].get("gaia_alignment"))
        )
        subsample_target = Subsample.from_yaml_dict(cfg[telescope_key]["target"])
        variable_covariance_config = variable_covariance_config_factory(
            cfg[telescope_key].get("variable_covariance")
        )

        return cls(
            zero_flux_prior_configs=zero_flux_prior_configs,
            rotation_and_shift_algorithm=rotation_and_shift_algorithm,
            psf_kernel_configs=psf_kernel_configs,
            data_loader=data_loader,
            subsample_target=subsample_target,
            extra_masks=extra_masks,
            star_alignment_config=star_alignment_config,
            variable_covariance_config=variable_covariance_config,
        )
