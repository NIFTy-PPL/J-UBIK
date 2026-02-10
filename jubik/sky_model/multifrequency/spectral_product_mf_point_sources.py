# Copyright(C) 2025
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

from functools import reduce
from typing import Optional, Union

import jax.numpy as jnp
import nifty.re as jft
from nifty.re.num.stats_distributions import normal_prior
from nifty.re.tree_math.vector import Vector
from numpy.typing import ArrayLike

from .spectral_product_utils.distribution_or_default import (
    build_distribution_or_default,
)
from .spectral_product_utils.frequency_deviations import (
    build_frequency_deviations_model_with_degeneracies,
)
from .spectral_product_utils.spectral_behavior import (
    SingleHarmonicLogSpectralBehavior,
)


class PointSourceSpectralIndex(SingleHarmonicLogSpectralBehavior):
    """Spectral index model for point sources.

    This is a reduced version of `SpectralIndex` without spatially correlated
    spectral fluctuations. It only contains a mean spectral index field; any
    frequency deviations are handled separately in `MultiFrequencyInvGammaSky`.
    """

    def __init__(
        self,
        log_frequencies: ArrayLike,
        mean: jft.Model,
        reference_frequency_index: int,
        spatial_shape: tuple[int, ...],
    ):
        """Point-source spectral index behavior.

        Parameters
        ----------
        log_frequencies: ArrayLike
            The log of the frequencies. The relative frequencies are calculated
            internally. See `reference_frequency_index`.
        mean: jft.Model
            The mean of the spectral index model.
        reference_frequency_index: int
            The index of the reference frequency. Used to compute the relative
            frequencies.
        spatial_shape: tuple[int, ...]
            Spatial shape of the point-source field. Used for broadcasting.
        """
        log_frequencies = jnp.array(log_frequencies)
        slicing_tuple = (slice(None),) + (None,) * len(spatial_shape)
        self._relative_log_frequencies = (
            log_frequencies - log_frequencies[reference_frequency_index]
        )[slicing_tuple]

        self._mean = mean
        self._spatial_shape = spatial_shape
        self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

        super().__init__(domain=self._mean.domain)

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    def mean(self, p) -> ArrayLike:
        return self._mean(p)

    def fluctuations(self, p) -> ArrayLike:
        return jnp.zeros(
            self._spatial_shape, dtype=self.relative_log_frequencies.dtype
        )

    def fluctuations_with_frequencies(self, p) -> ArrayLike:
        return self.fluctuations(p) * self.relative_log_frequencies

    def mean_with_frequencies(self, p) -> ArrayLike:
        return self.mean(p) * self.relative_log_frequencies

    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        dev_slope = (
            jnp.sum(deviations * self.relative_log_frequencies, axis=0)
            * self._denominator
        )
        return deviations - dev_slope * self.relative_log_frequencies

    def __call__(self, p):
        # Only needed for instantiation of the model.
        return None


class MultiFrequencyInvGammaSky(jft.Model):
    """A model for generating an uncorrelated multi-frequency point-source sky.

    The model applies a spectral behavior to an inverse-gamma distributed
    reference frequency map:

    .. math ::
        sky(\\nu) = i_\\mathrm{ref} \\cdot \\exp(
            \\mathrm{SpectralBehavior}(\\nu) + \\mathrm{deviations}(\\nu)
        )
    """

    def __init__(
        self,
        reference_frequency_model: jft.Model,
        log_spectral_behavior: SingleHarmonicLogSpectralBehavior,
        spectral_index_deviations: Optional[jft.Model] = None,
    ):
        """
        Parameters
        ----------
        reference_frequency_model: jft.Model
            Model for the reference-frequency point-source distribution.
        log_spectral_behavior: SingleHarmonicLogSpectralBehavior
            Log spectral behavior of the model.
        spectral_index_deviations: Optional[jft.Model]
            A model capturing deviations from the spectral behavior.
        """
        self.reference_frequency_model = reference_frequency_model
        self.log_spectral_behavior = log_spectral_behavior
        self.spectral_index_deviations = spectral_index_deviations

        models = [self.reference_frequency_model]

        logfreqs = self.log_spectral_behavior.relative_log_frequencies
        if logfreqs.shape[0] > 1:
            models += [
                self.log_spectral_behavior,
                self.spectral_index_deviations,
            ]

        domain = reduce(
            lambda a, b: a | b,
            [
                (m.domain.tree if isinstance(m.domain, Vector) else m.domain)
                for m in models
                if m is not None
            ],
        )

        self._has_spectral_axis = logfreqs.shape[0] > 1
        if not self._has_spectral_axis:
            self.spectral_index_deviations = None

        super().__init__(domain=domain)

    def __call__(self, p):
        if not self._has_spectral_axis:
            return self.reference_frequency_distribution(p)[None, ...]
        return self.reference_frequency_distribution(p) * jnp.exp(
            self.log_spectral_distribution(p)
        )

    def reference_frequency_distribution(self, p):
        """Convenience method to retrieve the reference-frequency distribution."""
        return self.reference_frequency_model(p)

    def log_spectral_distribution(self, p):
        """Convenience method to retrieve the log spectral distribution."""
        deviations = 0.0
        if self.spectral_index_deviations is not None:
            deviations = self.log_spectral_behavior.remove_degeneracy_of_spectral_deviations(
                self.spectral_index_deviations(p)
            )

        return (
            self.log_spectral_behavior.mean_with_frequencies(p)
            + self.log_spectral_behavior.fluctuations_with_frequencies(p)
            + deviations
        )

    def spectral_index_distribution(self, p):
        """Convenience method to retrieve the spectral index mean."""
        return self.log_spectral_behavior.mean(p)

    def spectral_deviations_distribution(self, p):
        """Convenience method to retrieve spectral deviations."""
        if self.spectral_index_deviations is None:
            return None
        return self.log_spectral_behavior.remove_degeneracy_of_spectral_deviations(
            self.spectral_index_deviations(p)
        )


def build_mf_invgamma_sky(
    prefix: str,
    alpha: float,
    q: float,
    shape: tuple[int, ...],
    log_frequencies: Union[tuple[float, ...], ArrayLike],
    reference_frequency_index: int,
    spectral_settings: dict,
    dtype: type = jnp.float64,
) -> MultiFrequencyInvGammaSky:
    """
    Builds a multi-frequency point-source sky model parametrized as

    .. math ::
        sky(\\nu) = i_\\mathrm{ref} \\cdot \\exp(
            \\alpha(\\nu-\\nu_\\mathrm{ref}) + \\mathrm{deviations}(\\nu)
        )

    Parameters
    ----------
    prefix: str
        Prefix for the model parameter names.
    alpha: float
        Shape parameter for the inverse-gamma prior.
    q: float
        Scale parameter for the inverse-gamma prior.
    shape: tuple[int, ...]
        Spatial shape of the point-source field.
    log_frequencies: Union[tuple[float, ...], ArrayLike]
        Logarithmic frequencies.
    reference_frequency_index: int
        Index of the reference frequency in `log_frequencies`.
    spectral_settings: dict
        Settings for the spectral index priors. Expected keys:
            - mean: callable or parameters (default normal prior)
            - deviations: dict, optional
            - shared or mean_shared: bool, optional (default False)
    dtype: type
        Data type of the parameters.

    Returns
    -------
    model: MultiFrequencyInvGammaSky
        The multi-frequency point-source sky model.
    """
    shared_mean = spectral_settings.get(
        "mean_shared", spectral_settings.get("shared", False)
    )
    mean_shape = () if shared_mean else shape
    log_frequencies = jnp.array(log_frequencies)

    reference_freq_model = build_distribution_or_default(
        (alpha, q),
        f"{prefix}_inv_gamma",
        jft.invgamma_prior,
        shape=shape,
        dtype=dtype,
    )

    spectral_index_mean = build_distribution_or_default(
        spectral_settings["mean"],
        f"{prefix}_spectral_index_mean",
        normal_prior,
        shape=mean_shape,
        dtype=dtype,
    )

    deviations_settings = spectral_settings.get("deviations")
    if log_frequencies.shape[0] <= 1:
        deviations_model = None
        if deviations_settings is not None:
            jft.logger.warning(
                "Ignoring spectral deviations for single-frequency inputs."
            )
    else:
        deviations_model = build_frequency_deviations_model_with_degeneracies(
            shape,
            log_frequencies,
            reference_frequency_index,
            deviations_settings,
            prefix=f"{prefix}_spectral",
        )

    spectral_index = PointSourceSpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        reference_frequency_index=reference_frequency_index,
        spatial_shape=shape,
    )

    return MultiFrequencyInvGammaSky(
        reference_freq_model,
        spectral_index,
        deviations_model,
    )
