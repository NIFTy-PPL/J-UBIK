import nifty8.re as jft

from functools import reduce

from .rotation_and_shift import build_rotation_and_shift_model
from .masking import build_mask
from .integration_model import build_sum_integration, build_sum_integration_old
from .rotation_and_shift import RotationAndShiftModel
from ..library.likelihood import build_gaussian_likelihood

from astropy.coordinates import SkyCoord
from typing import Tuple, Callable
from numpy.typing import ArrayLike


class DataModel(jft.Model):
    def __init__(
        self,
        sky_domain: dict,
        rotation_and_shift: RotationAndShiftModel,
        psf: Callable[ArrayLike, ArrayLike],
        integrate: Callable[ArrayLike, ArrayLike],
        mask: Callable[ArrayLike, ArrayLike]
    ):
        need_sky_key = 'Need to provide an internal key to the target of the sky model'
        assert isinstance(sky_domain, dict), need_sky_key

        self.rotation_and_shift = rotation_and_shift
        self.psf = psf
        self.integrate = integrate
        self.mask = mask

        super().__init__(
            domain=sky_domain | self.rotation_and_shift.domain
        )

    def __call__(self, x):
        return self.mask(self.integrate(self.psf(self.rotation_and_shift(x))))


def build_jwst_model(
    sky_model,
    reconstruction_grid,
    data_set,
    world_extrema_key: Tuple[SkyCoord]
):

    likelihoods = []
    for ii, (dkey, data) in enumerate(data_set.items()):

        rotation_and_shift = build_rotation_and_shift_model(
            sky_domain=sky_model.target,
            world_extrema_key=world_extrema_key,
            reconstruction_grid=reconstruction_grid,
            data_key=dkey,
            data_grid=data['grid'],
            **data,
        )

        # psf = build_psf_model(parameters)
        def psf(x): return x
        integrate = build_sum_integration(
            rotation_and_shift.target.shape,
            data['subsample'])
        # integrate = build_sum_integration_old(
        #     rotation_and_shift.target.shape,
        #     data['subsample'])
        mask = build_mask(data['mask'])

        data_model = DataModel(
            sky_model.target, rotation_and_shift, psf, integrate, mask)

        data['data_model'] = data_model

        likelihood = build_gaussian_likelihood(
            data['data'].reshape(-1), float(data['std']))
        likelihood = likelihood.amend(
            data_model, domain=jft.Vector(data_model.domain))
        likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x+y, likelihoods)
    return likelihood
