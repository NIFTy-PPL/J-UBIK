#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig

import jax.random as random
import pytest
from numpy.testing import assert_allclose
import numpy as np

import nifty8.re as jft
import jubik0 as ju

from nifty8.re.correlated_field import make_grid, NonParametricAmplitude

from jubik0.sky_model.multifrequency.spectral_product_utils.frequency_deviations import (
    build_frequency_deviations_model_with_degeneracies,
)
from jubik0.sky_model.multifrequency.spectral_product_utils.spectral_behavior import (
    SpectralIndex,
)
from jubik0.sky_model.multifrequency.spectral_product_utils.scaled_excitations import (
    build_scaled_excitations,
)

pmp = pytest.mark.parametrize


@pmp("shape", [(10,), (10, 10)])
@pmp("distances", [0.1])
@pmp(
    "log_frequencies",
    [
        np.array((0.1,)),
        np.array((0.01, 0.1, 1.0, 14.0)),
    ],
)
@pmp("reference_frequency_index", [0])
@pmp("zero_mode", [jft.Model(lambda p: 0.0, domain={"zero_mode": None})])
@pmp("spectral_index_mean", [jft.NormalPrior(0.0, 1.0, name="spectral_index_mean")])
@pmp(
    "spectral_index_fluctuations",
    [
        (0.5, 5.0e-1),
    ],
)
@pmp(
    "deviations_settings",
    [
        dict(
            process="wiener",
            sigma=(1.0, 0.1),
        ),
        None,
    ],
)
def test_correlated_multi_frequency_sky_init(
    shape,
    distances,
    log_frequencies,
    reference_frequency_index,
    zero_mode,
    spectral_index_mean,
    spectral_index_fluctuations,
    deviations_settings,
):
    grid = make_grid(shape, distances, "fourier")
    spatial_amplitude = NonParametricAmplitude(
        grid,
        None,
        jft.normal_prior(1.0, 1.0),
    )

    spatial_fluctuations = jft.LogNormalPrior(0.1, 10.0, name="spatial_fluctuations")

    scaled_excitations = build_scaled_excitations(
        prefix="spectral_index_",
        fluctuations_settings=spectral_index_fluctuations,
        shape=shape,
    )

    zero_mode = zero_mode
    spectral_behavior = SpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        spectral_scaled_excitations=scaled_excitations,
        reference_frequency_index=reference_frequency_index,
    )

    deviations_model = build_frequency_deviations_model_with_degeneracies(
        shape, log_frequencies, reference_frequency_index, deviations_settings
    )

    mf_sky = ju.SpectralProductSky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
        spectral_index_deviations=deviations_model,
    )

    assert mf_sky
    assert mf_sky.domain


@pmp("seed", [12, 42])
@pmp("shape", [(10,), (10, 10)])
@pmp("distances", [0.1])
@pmp("zero_mode", [jft.Model(lambda p: 0.1, domain={"test_zero_mode": None})])
@pmp("zero_mode_offset", [0.0])
@pmp("spectral_index_mean", [jft.NormalPrior(0.0, 1.0, name="spectral_index_mean")])
@pmp(
    "spectral_index_fluctuations",
    [jft.LogNormalPrior(0.1, 1.0, name="spectral_index_fluctuations")],
)
def test_spatial_convolution(
    seed,
    shape,
    distances,
    zero_mode,
    zero_mode_offset,
    spectral_index_mean,
    spectral_index_fluctuations,
):
    grid = make_grid(shape, distances, "fourier")
    prefix = "test"
    fluct = (1, 0.01)
    avgsl = (-4.0, 0.1)

    spatial_amplitude = NonParametricAmplitude(
        grid, None, jft.normal_prior(*avgsl), prefix=f"{prefix}_"
    )

    spatial_fluctuations = build_scaled_excitations(
        prefix=f"{prefix}_spatial",
        fluctuations_settings=fluct,
        shape=shape,
    )

    spectral_behavior = SpectralIndex(
        log_frequencies=np.array((0.0,)),
        mean=spectral_index_mean,
        spectral_scaled_excitations=spectral_index_fluctuations,
        reference_frequency_index=0,
    )

    mf_sky = ju.SpectralProductSky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
        nonlinearity=lambda x: x,
    )

    cfm = jft.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(zero_mode_offset, zero_mode)
    cfm.add_fluctuations(shape, distances, fluct, avgsl)
    cfm = cfm.finalize()
    rp = mf_sky.init(random.PRNGKey(seed))
    cfm_rp = {
        "zeromode": rp[f"{prefix}_zero_mode"],
        "fluctuations": rp[f"{prefix}_spatial_fluctuations"],
        "loglogavgslope": rp[f"{prefix}_loglogavgslope"],
        "xi": rp[f"{prefix}_spatial_xi"],
    }
    zm_index = tuple(0 for _ in grid.shape)
    # this happens because the cfm has a degeneracy in the
    # zero mode parametrization
    diff = zero_mode(rp) * (1 - rp[f"{prefix}_spatial_xi"][zm_index])
    assert_allclose(mf_sky(rp)[0] - diff, cfm(cfm_rp))


@pmp("seed", [12, 42])
@pmp("shape", [(10,), (10, 10)])
@pmp("distances", [0.1])
@pmp("log_frequencies", [(0.1, 0.2, 0.6)])
@pmp("zero_mode", [jft.Model(lambda p: 0.1, domain={"zero_mode": None})])
@pmp("spectral_index_mean", [jft.NormalPrior(0.0, 1.0, name="spectral_index_mean")])
@pmp("spectral_index_fluctuations", [(0.1, 1.0)])
def test_apply_with_and_without_frequency_deviations(
    seed,
    shape,
    distances,
    log_frequencies,
    zero_mode,
    spectral_index_mean,
    spectral_index_fluctuations,
):
    grid = make_grid(shape, distances, "fourier")
    reference_freq_idx = 0
    prefix = "test"
    fluct = (0.1, 0.01)
    avgsl = (-4.0, 0.1)
    spatial_amplitude = NonParametricAmplitude(
        grid,
        None,
        jft.normal_prior(*avgsl),
        prefix=f"{prefix}_",
    )

    log_frequencies = np.array(log_frequencies)

    spatial_fluctuations = build_scaled_excitations(
        prefix=f"{prefix}_spatial",
        fluctuations_settings=fluct,
        shape=shape,
    )
    spectral_fluctuations = build_scaled_excitations(
        prefix=f"{prefix}_spatial",
        fluctuations_settings=spectral_index_fluctuations,
        shape=shape,
    )

    spectral_behavior = SpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        spectral_scaled_excitations=spectral_fluctuations,
        reference_frequency_index=reference_freq_idx,
    )

    mf_sky_wo_deviations = ju.SpectralProductSky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
    )

    deviations_settings = dict(
        process="wiener",
        sigma=(1.0e-15, 1.0e-18),
    )

    deviations_model = build_frequency_deviations_model_with_degeneracies(
        shape, log_frequencies, 0, deviations_settings
    )

    mf_sky = ju.SpectralProductSky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
        spectral_index_deviations=deviations_model,
    )

    rp = mf_sky.init(random.PRNGKey(seed))
    assert_allclose(mf_sky_wo_deviations(rp), mf_sky(rp))


def test_simple_vs_complex_evaluation():
    from jubik0.sky_model.multifrequency.spectral_product_utils.spectral_behavior import (
        HarmonicLogSpectralBehavior,
    )
    import jax.numpy as jnp

    shape = (10, 10)
    distances = 0.1
    log_frequencies = (0.1, 0.2, 0.6)

    grid = make_grid(shape, distances, "fourier")
    reference_freq_idx = 0
    prefix = "test"
    fluct = (0.1, 0.01)
    avgsl = (-4.0, 0.1)
    spatial_amplitude = NonParametricAmplitude(
        grid,
        None,
        jft.normal_prior(*avgsl),
        prefix=f"{prefix}_",
    )

    log_frequencies = np.array(log_frequencies)

    spatial_fluctuations = build_scaled_excitations(
        prefix=f"{prefix}_spatial",
        fluctuations_settings=fluct,
        shape=shape,
    )

    spectral_index_mean = jft.NormalPrior(0.0, 1.0, name="spectral_index_mean")
    spectral_index_fluctuations = (0.1, 1.0)
    spectral_fluctuations = build_scaled_excitations(
        prefix=f"{prefix}_spatial",
        fluctuations_settings=spectral_index_fluctuations,
        shape=shape,
    )

    spectral_behavior = SpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        spectral_scaled_excitations=spectral_fluctuations,
        reference_frequency_index=reference_freq_idx,
    )
    zero_mode = jft.Model(lambda p: 0.1, domain={"zero_mode": None})

    class SpectralIndexWithoutSingleBehavior(HarmonicLogSpectralBehavior):
        def __init__(
            self,
            log_frequencies: np.ndarray,
            mean: jft.Model,
            spectral_scaled_excitations: jft.Model,
            reference_frequency_index: int,
        ):
            """Spectral Index spectral behavior. Special case of the
            `SpectralPolynomial`. However, since it has a single parameter, it can
            be used to speed up evaluations.

            .. math::
                \\log F(\\nu, A, ) = A^{spec.}
                * \\left\\{ \\mathrm{fluc.}^A \\xi^A \\nu  \\right\\}
                + \\mathrm{mean}^A \\nu

            Parameters
            ----------
            log_frequencies: ArrayLike
                The log of the frequencies. The relative frequencies are calculated
                internally. See `reference_frequency_index`.
            mean: Model
                The means of the polynomial model. See equation above.
            spectral_scaled_excitations: Model
                The fluctuations already applied to the xis. Hence, the
                combination of \\mathrm{fluc.}^A \\xi^A.
            reference_frequency_index: int
                The index of the reference frequency. Used in order to calculate
                the relative frequencies.
            """
            # NOTE : The model assumes that spectral axis is fixed to be the first
            # index of the output model. The slicing tuple matches the shape of the
            # relative_log_frequencies using the numpy slicing convention.
            slicing_tuple = (slice(None),) + (None,) * len(
                spectral_scaled_excitations.target.shape
            )

            self._relative_log_frequencies = (
                log_frequencies - log_frequencies[reference_frequency_index]
            )[slicing_tuple]

            self._mean = mean
            self.spectral_scaled_excitations = spectral_scaled_excitations

            self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

            super().__init__(
                domain=self._mean.domain | self.spectral_scaled_excitations.domain
            )

        @property
        def relative_log_frequencies(self):
            return self._relative_log_frequencies

        def mean(self, p):
            return self._mean(p)

        def fluctuations(self, p):
            return self.spectral_scaled_excitations(p)

        def fluctuations_with_frequencies(self, p):
            return self.fluctuations(p) * self.relative_log_frequencies

        def mean_with_frequencies(self, p):
            return self.mean(p) * self.relative_log_frequencies

        def remove_degeneracy_of_spectral_deviations(
            self, deviations: np.ndarray
        ) -> np.ndarray:
            dev_slope = (
                jnp.sum(deviations * self.relative_log_frequencies, axis=0)
                * self._denominator
            )
            return deviations - dev_slope * self.relative_log_frequencies

        def __call__(self, p):
            # NOTE : only needed for instantiation of the model. However, maybe
            # there is a better way !!!
            # Since the apply is not needed
            return None

    spectral_behavior_nonsingle = SpectralIndexWithoutSingleBehavior(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        spectral_scaled_excitations=spectral_fluctuations,
        reference_frequency_index=reference_freq_idx,
    )

    mf_sky_single = ju.SpectralProductSky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
    )
    mf_sky_nonsingle = ju.SpectralProductSky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior_nonsingle,
    )

    key = random.PRNGKey(52)
    ini = mf_sky_single.init(key)
    assert_allclose(mf_sky_single(ini), mf_sky_nonsingle(ini))
