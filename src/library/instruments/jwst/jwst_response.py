from typing import Callable, Optional

import nifty8.re as jft
from numpy.typing import ArrayLike

from .integration_model import build_sum_integration
from .jwst_psf import instantiate_psf, load_psf_kernel
from .rotation_and_shift import build_rotation_and_shift_model, \
    RotationAndShiftModel
from .zero_flux_model import build_zero_flux_model


class JwstResponse(jft.Model):
    """
    A that connects observational data to the corresponding sky and
    instrument models.

    This class models a data pipeline that includes rotation, shifting,
    PSF application, integration, transmission correction, and masking,
    with an optional zero-flux model.
    """

    def __init__(
        self,
        sky_domain: dict,
        rotation_and_shift: Optional[RotationAndShiftModel],
        psf: Callable[[ArrayLike], ArrayLike],
        integrate: Callable[[ArrayLike], ArrayLike],
        transmission: float,
        zero_flux_model: Optional[jft.Model],
        mask: Callable[[ArrayLike], ArrayLike]
    ):
        """
        Initialize the DataModel with components for various data
        transformations.

        Parameters
        ----------
        sky_domain : dict
            A dictionary defining the sky domain, with a single key
            corresponding to the internal target of the sky model.
            This defines the input space of the data.
        rotation_and_shift : RotationAndShiftModel, optional
            A model that applies rotation and shift transformations
            to the input data.
        psf : callable
            A function that applies a point spread function (PSF) to the
            input data.
        integrate : callable
            A function that performs integration on the input data.
        transmission : float
            A transmission factor by which the output data is multiplied.
        zero_flux_model : jft.Model, optional
            A secondary model to account for zero flux.
            If provided, its output is added to the domain model's output.
        mask : callable
            A function that applies a mask to the final output.

        Raises
        ------
        AssertionError
            If `sky_domain` is not a dictionary or if it contains
            more than one key.
        """
        need_sky_key = ('Need to provide an internal key to the target of the '
                        'sky model')
        assert isinstance(sky_domain, dict), need_sky_key
        assert len(sky_domain.keys()) == 1, need_sky_key

        self.rotation_and_shift = rotation_and_shift
        self.psf = psf
        self.integrate = integrate
        self.transmission = transmission
        self.zero_flux_model = zero_flux_model
        self.mask = mask

        domain = sky_domain | rotation_and_shift.domain
        if zero_flux_model is not None:
            domain = domain | zero_flux_model.domain
        super().__init__(domain=domain)

    def __call__(self, x):
        out = self.rotation_and_shift(x)
        out = self.psf(out)
        out = self.integrate(out)
        out = out * self.transmission
        if self.zero_flux_model is not None:
            out = out + self.zero_flux_model(x)
        out = self.mask(out)
        return out


def build_jwst_response(
    sky_domain: dict,
    subsample: int,
    rotation_and_shift_kwargs: Optional[dict],
    psf_kwargs: dict,
    transmission: float,
    data_mask: Optional[ArrayLike],
    zero_flux: Optional[dict],
) -> JwstResponse:
    """
    Builds the data model for a Jwst observation.

    The data model pipline:
    rotation_and_shift | psf | integrate | mask

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.

    subsample: int
        The subsample factor for the data grid.

    rotation_and_shift_kwargs: dict
        reconstruction_grid: Grid
        data_dvol: Unit, the volume of a data pixel
        data_wcs: WcsBase,
        data_model_type: str,
        kwargs_linear: dict, (order, sky_as_brightness, mode)
        kwargs_nufft: dict, (sky_as_brightness)
        kwargs_sparse: dict, (extend_factor, to_bottom_left)
        world_extrema: Tuple[SkyCoord]
        coordinate_correction: Optional[dict]
            domain_key: str
            priors: dict
                - shift: Mean and sigma for the Gaussian distribution
                of shift model.
                - rotation: Mean and sigma of the Gaussian distribution
                for theta [rad]

    psf_kwargs:
        camera: str, NIRCam or MIRI
        filter: str
        center_pix: tuple, pixel according to which to evaluate the psf model
        webbpsf_path: str
        fov_pixels: int, how many pixles considered for the psf,

    data_mask: ArrayLike
        The mask on the data
    """

    need_sky_key = ('Need to provide an internal key to the target of the sky '
                    'model.')
    assert isinstance(sky_domain, dict), need_sky_key

    rotation_and_shift = build_rotation_and_shift_model(
        sky_domain=sky_domain,
        reconstruction_grid=rotation_and_shift_kwargs['reconstruction_grid'],
        world_extrema=rotation_and_shift_kwargs['world_extrema'],
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

    integrate = build_sum_integration(
        high_res_shape=rotation_and_shift.target.shape,
        reduction_factor=subsample,
    )

    psf_kernel = load_psf_kernel(
        camera=psf_kwargs['camera'],
        filter=psf_kwargs['filter'],
        center_pixel=psf_kwargs['center_pixel'],
        webbpsf_path=psf_kwargs['webbpsf_path'],
        psf_library_path=psf_kwargs['psf_library_path'],
        fov_pixels=psf_kwargs.get('fov_pixels'),
        fov_arcsec=psf_kwargs.get('fov_arcsec'),
        subsample=subsample,
    ) if len(psf_kwargs) != 0 else None
    psf = instantiate_psf(psf_kernel)

    if zero_flux is None:
        zero_flux_model = None
    else:
        zero_flux_model = build_zero_flux_model(zero_flux['dkey'], zero_flux)

    if data_mask is None:
        def mask(x): return x
    else:
        def mask(x): return x[data_mask]

    return JwstResponse(sky_domain=sky_domain,
                        rotation_and_shift=rotation_and_shift,
                        psf=psf,
                        integrate=integrate,
                        transmission=transmission,
                        zero_flux_model=zero_flux_model,
                        mask=mask)
