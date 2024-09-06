import nifty8.re as jft

from .rotation_and_shift import build_rotation_and_shift_model
from .masking import build_mask
from .psf.build_psf import instantiate_psf, load_psf_kernel
from .integration_model import build_sum_integration
from .rotation_and_shift import RotationAndShiftModel
from .zero_flux_model import build_zero_flux_model

from .reconstruction_grid import Grid
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
        transmission: float,
        zero_flux_model: jft.Model,
        mask: Callable[ArrayLike, ArrayLike]
    ):
        need_sky_key = ('Need to provide an internal key to the target of the '
                        'sky model')
        assert isinstance(sky_domain, dict), need_sky_key

        self.rotation_and_shift = rotation_and_shift
        self.psf = psf
        self.integrate = integrate
        self.transmission = transmission
        self.zero_flux_model = zero_flux_model
        self.mask = mask

        super().__init__(
            domain=sky_domain | self.rotation_and_shift.domain | self.zero_flux_model.domain
        )

    def __call__(self, x):
        out = self.rotation_and_shift(x)
        out = self.psf(out)
        out = self.integrate(out)
        out = out * self.transmission
        out = out + self.zero_flux_model(x)
        out = self.mask(out)
        return out


def build_data_model(
    sky_domain: dict,
    reconstruction_grid: Grid,
    subsample: int,
    rotation_and_shift_kwargs: dict,
    psf_kwargs: dict,
    transmission: float,
    data_mask: ArrayLike,
    world_extrema: Tuple[SkyCoord],
    zero_flux: dict,
) -> DataModel:
    '''Build the data model for a Jwst observation. The data model pipline:
    rotation_and_shift | psf | integrate | mask

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.

    reconstruction_grid: Grid

    subsample: int
        The subsample factor for the data grid.

    rotation_and_shift_kwargs: dict
        data_dvol: Unit, the volume of a data pixel
        data_wcs: WcsBase,
        data_model_type: str,
        kwargs_linear: dict, (order, sky_as_brightness, mode)
        kwargs_nufft: dict, (sky_as_brightness)
        kwargs_sparse: dict, (extend_factor, to_bottom_left)
        coordinate_correction: Optional[dict]
            domain_key: str
            priors: dict
                - shift: Mean and sigma for the Gaussian distribution of shift model.
                - rotation: Mean and sigma of the Gaussian distribution for theta [rad]

    psf_kwargs:
        camera: str, NIRCam or MIRI
        filter: str
        center_pix: tuple, pixel according to which to evaluate the psf model
        webbpsf_path: str
        fov_pixels: int, how many pixles considered for the psf,

    data_mask: ArrayLike
        The mask on the data

    world_extrema: Tuple[SkyCoord]
        The extrema for the evaluation
    '''

    need_sky_key = ('Need to provide an internal key to the target of the sky '
                    'model.')
    assert isinstance(sky_domain, dict), need_sky_key

    rotation_and_shift = build_rotation_and_shift_model(
        sky_domain=sky_domain,
        world_extrema=world_extrema,
        reconstruction_grid=reconstruction_grid,
        data_grid_dvol=rotation_and_shift_kwargs['data_dvol'],
        data_grid_wcs=rotation_and_shift_kwargs['data_wcs'],
        model_type=rotation_and_shift_kwargs['data_model_type'],
        subsample=subsample,
        kwargs=dict(
            linear=rotation_and_shift_kwargs.get(
                'kwargs_linear', dict(order=1, sky_as_brightness=False)),
            nufft=rotation_and_shift_kwargs.get(
                'kwargs_nufft', dict(sky_as_brightness=False)),
            sparse=rotation_and_shift_kwargs.get(
                'kwargs_sparse', dict(extend_factor=1, to_bottom_left=True)),
        ),
        coordinate_correction=rotation_and_shift_kwargs.get(
            'shift_and_rotation_correction', None)
    )

    psf_kernel = load_psf_kernel(
        camera=psf_kwargs['camera'],
        filter=psf_kwargs['filter'],
        center_pixel=psf_kwargs['center_pixel'],
        webbpsf_path=psf_kwargs['webbpsf_path'],
        psf_library_path=psf_kwargs['psf_library_path'],
        fov_pixels=psf_kwargs['fov_pixels'],
        subsample=subsample,
    ) if len(psf_kwargs) != 0 else None
    psf = instantiate_psf(psf_kernel)

    integrate = build_sum_integration(
        high_res_shape=rotation_and_shift.target.shape,
        reduction_factor=subsample,
    )

    zero_flux_model = build_zero_flux_model(zero_flux['dkey'], zero_flux)

    mask = build_mask(data_mask)

    return DataModel(sky_domain=sky_domain,
                     rotation_and_shift=rotation_and_shift,
                     psf=psf,
                     integrate=integrate,
                     transmission=transmission,
                     zero_flux_model=zero_flux_model,
                     mask=mask)
