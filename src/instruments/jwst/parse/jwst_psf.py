from dataclasses import dataclass


WEBBPSF_PATH_KEY = 'webbpsf_path'
PSF_LIBRARY_PATH_KEY = 'psf_library'
PSF_ARCSEC_KEY = 'psf_arcsec_extension'
NORMALIZE_KEY = 'normalize'
NORMALIZE_DEFAULT = 'last'


@dataclass
class PsfKernelConfig:
    '''The PsfKernelConfig is a data model for holding metadata for
    the evaluation of the psf kernel.

    webbpsf_path : str
        The path to the directory containing the `webbpsf` data files.
    psf_library_path : str
        The directory where the computed PSF files are stored.
        The PSF kernel will be saved here if it is not already present.
    psf_arcsec : float
        The size of the PSF evaluation in arcsec. If not provided, `psf_pixels`
        must be specified.
    normalize : str
        The normalization method for the PSF. Default is 'last',
        but other methods may be supported by `webbpsf`.
    '''
    webbpsf_path: str
    psf_library_path: str
    psf_arcsec: float
    normalize: str


def yaml_to_psf_kernel_config(
    psf_config: dict,
):
    '''Read the PsfKernelConfig from the yaml config.

    psf_config: dict
        The dictionary holding:
            - webbpsf_path
            - psf_library_path
            - psf_arcsec_extension
            - normalize (Optional)
    '''

    return PsfKernelConfig(
        webbpsf_path=psf_config[WEBBPSF_PATH_KEY],
        psf_library_path=psf_config[PSF_LIBRARY_PATH_KEY],
        psf_arcsec=psf_config[PSF_ARCSEC_KEY],
        normalize=psf_config.get(NORMALIZE_KEY, NORMALIZE_DEFAULT),
    )
