from dataclasses import dataclass, field, replace
from typing import Callable, Union

import nifty.re as jft
from jax import Array
from numpy.typing import NDArray

from ....grid import Grid
from ..data.observation import Observation
from ..mosaicing.sky_beamer import SkyBeamer
from ..noise.factory_noise_correction import factory_noise_correction_model
from ..parse.noise.base_line_correction import BaseLineCorrection
from ..parse.response import Ducc0Settings, FinufftSettings
from ..response import _hermitian_adjoint, interferometry_response


def create_response_operator(
    domain: dict,
    sky2vis: Callable[[Array], Array],
    field_name: str,
    shift: jft.Model | None = None,
):
    """Create the full response operator.

    The response operator consists of the following pipeline
    1. Field extraction
    2. sky2vis
    3. shift (optional)

    Parameters
    ----------
    domain: dict,
        The domain has to contain the `field_name`.
    sky2vis: InterferometryResponse,
        FFT and Gridding
    field_name: str,
        The name of the field to be extracted from the (beam corrected) sky.
    shift: jft.Model | None = None,
        (Optional) multiplicative correction applied to the visibilities, e.g. a
        phase-shift correction.
    """

    response = jft.wrap(sky2vis, field_name)

    if shift is not None:
        domain = domain | shift.domain
        return jft.Model(lambda x: shift(x) * response(x), domain=domain)

    return jft.Model(response, domain=domain)


@dataclass
class LikelihoodBuilderBase:
    """Builder for a radio likelihood

    Attributes
    ----------
    visibilities: NDArray
        The visibilities of the observation.
    weight: NDArray
        The weights of the visibilities of the observation.
    response: jft.Model
        The response operator.
    name: str = "lh_{index}"
        (Optional) string for display in the minimization.
        This will only be displayed when having multiple likelihoods.
    shift: jft.Model | None = None
        (Optional) the phase factor inside `response`, see `build_phase_shift`.
    gain: jft.Model | None = None
        (Optional) the gain factor inside `response`, see `with_gain`.
    """

    response: jft.Model
    observation: Observation
    field_name: str
    shift: jft.Model | None = field(default=None, kw_only=True)
    gain: jft.Model | None = field(default=None, kw_only=True)

    def with_gain(self, gain: jft.Model) -> "LikelihoodBuilderBase":
        """A copy of this builder whose response is `gain(x) * response(x)`.

        The likelihood of the copy takes the gain latents next to the sky
        latents. The builder does not build the gain: the caller passes any
        model of the antenna gains, e.g. on top of `CalibrationDistributor`.

        Parameters
        ----------
        gain: jft.Model
            The complex factor of every visibility, shape and dtype of
            `visibilities` or broadcastable to them.

        Raises
        ------
        ValueError
            If the builder already has a gain.
        """
        if self.gain is not None:
            raise ValueError(f"{self.field_name}: the likelihood already has a gain.")
        response = self.response
        return replace(
            self,
            response=jft.Model(
                lambda x: gain(x) * response(x),
                domain=response.domain | gain.domain,
            ),
            gain=gain,
        )

    def response_adjoint(
        self, primals: NDArray | Array, x: dict | jft.Vector | None = None
    ) -> dict[str, Array]:
        """Apply the Hermitian adjoint of the response to `primals`.

        With a shift or a gain the response is nonlinear in them, so its
        adjoint is taken in the sky with both held fixed at the latent sample
        `x`, i.e. `L^H(conj(g(x) * p(x)) * primals)`. Without either `x` is
        ignored.
        """
        fixed = [m for m in (self.shift, self.gain) if m is not None]
        if not fixed:
            return _hermitian_adjoint(self.response, self.response.domain, primals)
        if x is None:
            raise ValueError(
                "The response has a shift or a gain: pass the latent sample `x` "
                "whose shift and gain to hold fixed."
            )
        theta = {k: x[k] for m in fixed for k in m.domain}
        sky_domain = {k: v for k, v in self.response.domain.items() if k not in theta}
        return _hermitian_adjoint(
            lambda sky: self.response(sky | theta), sky_domain, primals
        )

    @property
    def visibilities(self) -> NDArray:
        return self.observation.vis.val.val

    @property
    def weight(self) -> NDArray:
        return self.observation.weight.val.val

    @property
    def uvw(self) -> NDArray:
        return self.observation.uvw


@dataclass
class LikelihoodBuilder(LikelihoodBuilderBase):
    @property
    def likelihood(self) -> jft.Likelihood:
        likelihood = jft.Gaussian(
            self.visibilities, noise_cov_inv=lambda x: x * self.weight
        )
        return likelihood.amend(self.response, domain=jft.Vector(self.response.domain))

    def calibration_likelihood(
        self, model_visibilities: NDArray | Array
    ) -> jft.Likelihood:
        """The likelihood of the gain alone, for fixed model visibilities.

        Parameters
        ----------
        model_visibilities: NDArray | Array
            The model visibilities without the gain, shape
            `(n_pol, n_rows, n_freq)`.

        Raises
        ------
        ValueError
            If the builder has no gain, see `with_gain`.
        """
        if self.gain is None:
            raise ValueError("The likelihood has no gain to calibrate.")
        model = jft.Model(
            lambda x: self.gain(x) * model_visibilities, domain=self.gain.domain
        )
        likelihood = jft.Gaussian(
            self.visibilities, noise_cov_inv=lambda x: x * self.weight
        )
        return likelihood.amend(model, domain=jft.Vector(model.domain))


@dataclass
class VariableLikelihoodBuilder(LikelihoodBuilderBase):
    inverse_standard_deviation: jft.Model

    @property
    def likelihood(self) -> jft.Likelihood:
        model = jft.Model(
            lambda x: (self.response(x), self.inverse_standard_deviation(x)),
            domain=self.response.domain | self.inverse_standard_deviation.domain,
        )
        likelihood = jft.VariableCovarianceGaussian(self.visibilities)
        return likelihood.amend(model, domain=jft.Vector(model.domain))

    def calibration_likelihood(
        self, model_visibilities: NDArray | Array
    ) -> jft.Likelihood:
        """The likelihood of the gain and the noise alone, for fixed model
        visibilities, see `LikelihoodBuilder.calibration_likelihood`."""
        if self.gain is None:
            raise ValueError("The likelihood has no gain to calibrate.")
        model = jft.Model(
            lambda x: (
                self.gain(x) * model_visibilities,
                self.inverse_standard_deviation(x),
            ),
            domain=self.gain.domain | self.inverse_standard_deviation.domain,
        )
        likelihood = jft.VariableCovarianceGaussian(self.visibilities)
        return likelihood.amend(model, domain=jft.Vector(model.domain))


def build_likelihood_from_sky_beamer(
    observation: Observation,
    field_name: str,
    sky_beamer: SkyBeamer,
    sky_grid: Grid,
    backend_settings: Union[Ducc0Settings, FinufftSettings],
    noise_std_correction: BaseLineCorrection | None = None,
    shift: jft.Model | None = None,
) -> LikelihoodBuilder | VariableLikelihoodBuilder:
    """Create a likelihood builder corresponding to the `field_name`.

    The building consists of two steps:
    1. Build response operator

    2. Build the

    First, builds response operator, which takes the `field_name` from the
    sky_beamer operator and calculates the visibilities corresponding to the
    observation.
    Second, builds the likelihood operator corresponding to this observation.

    Parameters
    ----------
    observation: Observation
        The observation under question
    field_name: str
        The name of the field (pointing) corresponding to the `observation`.
    sky_beamer: SkyBeamer
        The operator that applies the beam pattern to the sky.
        This can potentially hold multiple pointings, that are identified by different
        field_name.
    sky_grid: Grid
        Used for building the InterferometryResponse
    backend_settings: Union[Ducc0Settings, FinufftSettings]
        The algorithm for gridding and fft.
    noise_std_correction: BaseLineCorrection | None
        (Optional) settings for an inferred noise standard deviation. If given,
        a `VariableLikelihoodBuilder` is returned instead of a
        `LikelihoodBuilder`.
    shift: jft.Model | None
        (Optional) phase factor multiplied onto the model visibilities, see
        `build_phase_shift`.

    Returns
    ------
    LikelihoodBuilder
        The likelihood corresponding to the `observation` and the `field_name`.
    """

    sky2vis = interferometry_response(
        observation=observation,
        sky_grid=sky_grid,
        backend_settings=backend_settings,
    )

    response = create_response_operator(
        domain=sky_beamer.target,
        sky2vis=sky2vis,
        field_name=field_name,
        shift=shift,
    )

    if noise_std_correction is None:
        return LikelihoodBuilder(
            observation=observation,
            response=response,
            field_name=field_name,
            shift=shift,
        )
    else:
        inverse_std_model = factory_noise_correction_model(
            correction_settings=noise_std_correction,
            observation=observation,
            prefix=f"{field_name}_noise_std",
        )
        return VariableLikelihoodBuilder(
            observation=observation,
            response=response,
            field_name=field_name,
            inverse_standard_deviation=inverse_std_model,
            shift=shift,
        )
