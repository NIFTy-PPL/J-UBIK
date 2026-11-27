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
from .masking.data_mask import CornerMasks, NanMasks
from .data.preloader.data_bounds import DataBoundsAdjust


@dataclass
class ConfigParserJwst:
    zero_flux_prior_configs: ZeroFluxPriorConfigs
    calibration_prior_configs: ZeroFluxPriorConfigs | None
    rotation_and_shift_algorithm: LinearConfig | NufftConfig
    psf_kernel_configs: JwstPsfKernelConfig
    data_loader: DataLoadingConfig
    subsample_target: Subsample
    corner_masks: CornerMasks
    nan_masks: NanMasks
    star_alignment_config: StarAlignmentConfig | None
    variable_covariance_config: VariableCovarianceConfig | None
    data_bounds_adjust: DataBoundsAdjust
    psf_broadening: ZeroFluxPriorConfigs | None = None

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
        filters = {f.lower() for f in data_loader.paths}
        for key in ("correction", "brighter_fatter", "joint_stars"):
            if unknown := set(getattr(psf_kernel_configs, key)) - filters:
                raise ValueError(
                    f"psf {key}: {sorted(unknown)} match no filter of "
                    f"{sorted(filters)}"
                )
        # one factor per exposure, the zero_flux layout: a default and filter keys
        calibration = ZeroFluxPriorConfigs.from_yaml_dict(
            cfg[telescope_key].get("calibration")
        )
        # a mistyped filter key would silently fall back to the default prior
        if unknown := set(calibration.names if calibration else {}) - filters:
            raise ValueError(
                f"calibration: {sorted(unknown)} match no filter of {sorted(filters)}"
            )
        corner_masks = CornerMasks.from_yaml_dict(cfg[telescope_key])

        star_alignment_config: StarAlignmentConfig | None = (
            StarAlignmentConfig.from_yaml_dict(cfg[telescope_key].get("gaia_alignment"))
        )
        subsample_target = Subsample.from_yaml_dict(cfg[telescope_key]["target"])
        if psf_kernel_configs.joint_stars and star_alignment_config is not None:
            raise ValueError(
                "psf joint_stars: the gaia_alignment stars cannot take the learned "
                "correction; drop one"
            )
        # the stars' kernels get the correction too, on its grid
        if psf_kernel_configs.correction and star_alignment_config is not None:
            if star_alignment_config.subsample != subsample_target:
                raise ValueError(
                    "psf correction: gaia_alignment.subsample "
                    f"({star_alignment_config.subsample}) must equal target.subsample "
                    f"({subsample_target}), the subsample the correction is for"
                )
        variable_covariance_config = variable_covariance_config_factory(
            cfg[telescope_key].get("variable_covariance")
        )

        nan_masks = NanMasks.from_yaml_dict(cfg[telescope_key])
        data_bounds_adjust = DataBoundsAdjust.from_yaml_dict(
            cfg[telescope_key].get("adjust_data_bounds", {})
        )
        # the σ (arcsec) of each exposure's Gaussian PSF broadening: a default and
        # filter keys, as zero_flux
        psf_broadening = ZeroFluxPriorConfigs.from_yaml_dict(
            cfg[telescope_key].get("psf_broadening")
        )
        # a mistyped filter key would silently fall back to the default prior
        if unknown := set(psf_broadening.names if psf_broadening else {}) - filters:
            raise ValueError(
                f"psf_broadening: {sorted(unknown)} match no filter of "
                f"{sorted(filters)}"
            )
        if psf_broadening and (
            both := [
                f
                for f in psf_kernel_configs.joint_stars
                if psf_broadening.get_name_setting_or_default(f) is not None
            ]
        ):
            raise ValueError(
                f"psf_broadening and psf joint_stars both broaden {both}; the "
                "learned correction has the width"
            )

        return cls(
            zero_flux_prior_configs=zero_flux_prior_configs,
            calibration_prior_configs=calibration,
            rotation_and_shift_algorithm=rotation_and_shift_algorithm,
            psf_kernel_configs=psf_kernel_configs,
            data_loader=data_loader,
            subsample_target=subsample_target,
            corner_masks=corner_masks,
            nan_masks=nan_masks,
            star_alignment_config=star_alignment_config,
            variable_covariance_config=variable_covariance_config,
            data_bounds_adjust=data_bounds_adjust,
            psf_broadening=psf_broadening,
        )
