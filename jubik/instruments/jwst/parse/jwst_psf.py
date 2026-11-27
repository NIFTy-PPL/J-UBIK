from dataclasses import dataclass, field

import numpy as np

WEBBPSF_PATH_KEY = "webbpsf_path"
PSF_LIBRARY_PATH_KEY = "psf_library"
PSF_ARCSEC_KEY = "psf_arcsec_extension"
NORMALIZE_KEY = "normalize"
NORMALIZE_DEFAULT = "last"
OPD_BY_DATE_KEY = "opd_by_date"
CORRECTION_KEY = "correction"
PIXEL_BOX_KEY = "pixel_box"
BRIGHTER_FATTER_KEY = "brighter_fatter"


@dataclass
class JwstPsfKernelConfig:
    """The PsfKernelConfig is a data model for holding metadata for
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
    opd_by_date : bool
        If True, use the telescope OPD measured by the JWST Wavefront Sensing
        System closest to each exposure's start, instead of STPSF's default
        OPD. The first run needs MAST network access. Default is False.
    correction : dict[str, str]
        Filter name (lower case) -> a `PsfCorrection` file (`PsfCorrection.save`),
        the learned correction of that filter's kernels, `fit_psf_correction` on
        the field's stars. Every kernel of the filter, the target's and the Gaia
        stars', becomes `PsfCorrection.apply(kernel)`: STPSF's kernel at its
        position times the shared texture, rescaled to its sum, then broadened.
        The file must be for the target's `subsample`. Filters without an entry
        keep STPSF's kernel. Default: none. Config::

            psf:
              opd_by_date: true          # the base the correction was learned on
              correction:
                f277w: psf_correction/f277w.npz
    pixel_box : bool
        If True, every kernel (after its correction) is convolved with a box
        one kernel pixel (sub-pixel) wide, `with_pixel_box`: the sum of the
        subsampled pixels is then the exact pixel integral, where the sky is
        band-limited (``rotation_and_shift.nufft.band_limit``) and the band
        plus the kernel's stays below the subsampled sampling frequency.
        Without it the sum is sharper by one sub-pixel box. A correction must
        have been learned the same way (`fit_psf_correction(pixel_box=...)`,
        on kernels loaded without the box: the fit adds it).
        Default is False.
    brighter_fatter : dict[str, float]
        Filter name (lower case) -> ``kappa`` per DN per integration of the
        brighter-fatter effect, `charge_migration`: the target response moves
        charge between the data pixels of each exposure's model in DN per
        integration (EFFINTTM / PHOTMJSR per MJy/sr), so every pixel broadens
        by its own brightness instead of one width for all. A correction of the
        filter must have been learned with the same kappa
        (`fit_psf_correction(brighter_fatter=...)`). Filters without an entry
        move no charge. The gaia_alignment stars move none either: they get
        the correction's width alone, too narrow for bright stars (their
        shifts stay unbiased). Default: none. Config::

            psf:
              brighter_fatter:
                f277w: 9.02e-7
    """

    webbpsf_path: str
    psf_library_path: str
    psf_arcsec: float
    normalize: str
    opd_by_date: bool = False
    correction: dict[str, str] = field(default_factory=dict)
    pixel_box: bool = False
    brighter_fatter: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml_dict(cls, raw: dict):
        """Read the PsfKernelConfig from the yaml config.

        Parameters
        ----------
        raw: dict, Parsed dict from yaml file, containing:
            - webbpsf_path
            - psf_library_path
            - psf_arcsec_extension
            - normalize | None
            - opd_by_date | None
            - correction | None
            - pixel_box | None
            - brighter_fatter | None
        """
        kappa = raw.get(BRIGHTER_FATTER_KEY) or {}
        if not isinstance(kappa, dict):
            raise ValueError(f"psf brighter_fatter must map filter -> kappa: {kappa}")
        kappa = {k.lower(): float(v) for k, v in kappa.items()}
        if bad := {k: v for k, v in kappa.items() if not 0 <= v < np.inf}:
            raise ValueError(f"psf brighter_fatter must be finite and >= 0: {bad}")

        return JwstPsfKernelConfig(
            webbpsf_path=raw[WEBBPSF_PATH_KEY],
            psf_library_path=raw[PSF_LIBRARY_PATH_KEY],
            psf_arcsec=raw[PSF_ARCSEC_KEY],
            normalize=raw.get(NORMALIZE_KEY, NORMALIZE_DEFAULT),
            opd_by_date=raw.get(OPD_BY_DATE_KEY, False),
            correction={
                k.lower(): v for k, v in (raw.get(CORRECTION_KEY) or {}).items()
            },
            pixel_box=raw.get(PIXEL_BOX_KEY, False),
            brighter_fatter=kappa,
        )
