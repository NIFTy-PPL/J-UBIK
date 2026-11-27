from dataclasses import dataclass, field, replace

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
JOINT_STARS_KEY = "joint_stars"


@dataclass(frozen=True)
class StarExclusion:
    """``psf.joint_stars``' ``exclude``: rules that drop stars from the fit, each
    off at None (or False). A stamp or a Gaia cutout (one star, one exposure)
    goes when

    max_peak : float | None
        its peak above its edge exceeds this many DN per integration (the star
        fit's ``--max-peak``; a stamp's ``fluence``);
    saturated : bool
        it is a Gaia cutout with DQ SATURATED within 2 pixels of the star's
        catalog pixel (psf_check's rule for the stamps), or that pixel masked.

    A Gaia star goes, with every stamp at it, when its Gaia ``ruwe``,
    ``astrometric_excess_noise_sig`` or ``classprob_dsc_combmod_galaxy``
    exceeds ``ruwe``, ``excess_noise_sig`` or ``galaxy`` (a binary, a poor
    astrometric fit, a galaxy); a column the Gaia table lacks skips its rule.
    """

    max_peak: float | None = 30000.0
    saturated: bool = True
    ruwe: float | None = 1.4
    excess_noise_sig: float | None = 2.0
    galaxy: float | None = 0.5


@dataclass(frozen=True)
class JointStars:
    """One filter's ``psf.joint_stars`` entry, see `JwstPsfKernelConfig`.

    stamps : str
        A `save_star_stamps` file: the stars and their base kernels.
    radius, fluctuations, loglogavgslope, width, fluence, orthogonal
        `fit_psf_correction`'s options, with its defaults.
    brighter_fatter : tuple[float, float] | None
        Mean and std of a log-normal prior of kappa per DN per integration,
        learned; None: kappa is ``psf.brighter_fatter``'s, or none.
    exclude : StarExclusion | None
        The stars to drop (``exclude: true`` for the defaults, or a map of
        the rules to change); None (the default) drops none.
    """

    stamps: str
    radius: int | None = None
    fluctuations: float = 0.1
    loglogavgslope: float = -2.0
    width: tuple[float, float] | None = None
    fluence: float | None = None
    orthogonal: bool = False
    brighter_fatter: tuple[float, float] | None = None
    exclude: StarExclusion | None = None


def _exclusion(raw) -> StarExclusion | None:
    """``exclude``: None or False, none; True, the defaults; a map, the
    defaults with its rules, numbers or None (off)."""
    if raw is None or raw is False:
        return None
    if raw is True:
        return StarExclusion()
    rules = StarExclusion(**raw)
    numbers = {
        k: None if getattr(rules, k) is None else float(getattr(rules, k))
        for k in ("max_peak", "ruwe", "excess_noise_sig", "galaxy")
    }
    return replace(rules, saturated=bool(rules.saturated), **numbers)


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
        (`fit_psf_correction(brighter_fatter=...)`). With ``joint_stars`` the
        stars move charge by it too, the gaia_alignment stars included.
        Filters without an entry move no charge. Without ``joint_stars`` the
        gaia_alignment stars move none: they get the correction's width alone,
        too narrow for bright stars (their shifts stay unbiased). Default:
        none. Config::

            psf:
              brighter_fatter:
                f277w: 9.02e-7
    joint_stars : dict[str, JointStars]
        Filter name (lower case) -> the correction of ``correction``, learned in
        the lens fit together with the stars of a `save_star_stamps` file
        instead of read from a file (`joint_correction`): the target's kernels
        and the stars' share the texture, the width (at ``fluence`` with one)
        and, with a ``brighter_fatter`` prior, kappa, and the stars'
        likelihood joins the lens's. The stamps' kernels must be for the
        target's ``subsample``, loaded as the target's are but without the box
        (``pixel_box`` applies to both). A filter in ``joint_stars`` has no
        ``correction`` nor ``psf_broadening``; kappa is given here or in
        ``brighter_fatter``, not both.

        With ``gaia_alignment`` (at the target's ``subsample``) every star is
        modelled once, by the same exact model: a stamp at a Gaia star (by its
        ``exposure``, ``ra`` and ``dec`` in the stamps file, within the shift
        prior's reach plus a pixel; ``x`` and ``y`` place it in the file) sits
        at Gaia's position, moved by its exposure's shift and, with
        ``fit_star_positions``, its astrometry; a Gaia star without a stamp in
        an exposure is its gaia_alignment cutout, placed the same way, with its
        own flux and background; the other stamps keep free positions. The
        filter then has no bilinear alignment term. Stars with a Gaia parallax
        of 1 mas or more are logged: their positions leave it out. An
        ``exclude`` entry drops stars by rules (`StarExclusion`: saturated or
        too bright frames, Gaia's RUWE, excess noise and galaxy probability),
        each drop logged with its reason. Default: none. Config::

            psf:
              pixel_box: true
              joint_stars:
                f277w:
                  stamps: psf_correction/b1938_f277w_stars.npz
                  radius: 30
                  width: [0.4, 0.3]
                  orthogonal: true
                  brighter_fatter: [9.02e-7, 2.0e-7]  # kappa learned
                  exclude: true  # or e.g. {ruwe: 1.6, galaxy: null}
    """

    webbpsf_path: str
    psf_library_path: str
    psf_arcsec: float
    normalize: str
    opd_by_date: bool = False
    correction: dict[str, str] = field(default_factory=dict)
    pixel_box: bool = False
    brighter_fatter: dict[str, float] = field(default_factory=dict)
    joint_stars: dict[str, JointStars] = field(default_factory=dict)

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
            - joint_stars | None
        """
        kappa = raw.get(BRIGHTER_FATTER_KEY) or {}
        if not isinstance(kappa, dict):
            raise ValueError(f"psf brighter_fatter must map filter -> kappa: {kappa}")
        kappa = {k.lower(): float(v) for k, v in kappa.items()}
        if bad := {k: v for k, v in kappa.items() if not 0 <= v < np.inf}:
            raise ValueError(f"psf brighter_fatter must be finite and >= 0: {bad}")
        correction = {k.lower(): v for k, v in (raw.get(CORRECTION_KEY) or {}).items()}
        joint = {}
        for name, entry in (raw.get(JOINT_STARS_KEY) or {}).items():
            name = name.lower()
            try:
                entry = JointStars(**entry)
                pairs = {k: getattr(entry, k) for k in ("width", "brighter_fatter")}
                # numbers, as for psf brighter_fatter: yaml reads 1e-6 as a string
                joint[name] = replace(
                    entry,
                    exclude=_exclusion(entry.exclude),
                    fluctuations=float(entry.fluctuations),
                    loglogavgslope=float(entry.loglogavgslope),
                    fluence=None if entry.fluence is None else float(entry.fluence),
                    **{
                        k: None if v is None else tuple(map(float, v))
                        for k, v in pairs.items()
                    },
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"psf joint_stars {name}: {e}") from e
            if joint[name].radius is not None and joint[name].radius < 1:
                raise ValueError(
                    f"psf joint_stars {name}: radius must be >= 1; omit it for no "
                    "texture"
                )
            prior = joint[name].brighter_fatter
            if prior is not None and not (
                np.shape(prior) == (2,)
                and np.all(np.isfinite(prior) & (np.asarray(prior) > 0))
            ):
                raise ValueError(
                    f"psf joint_stars {name}: brighter_fatter must be a log-normal "
                    f"[mean, std] > 0: {prior}"
                )
            if prior is not None and name in kappa:
                raise ValueError(
                    f"psf joint_stars {name}: kappa is learned here and fixed in "
                    "psf brighter_fatter; give one"
                )
            if name in correction:
                raise ValueError(
                    f"psf joint_stars {name}: the filter's correction is learned, "
                    "drop its psf correction file"
                )

        return JwstPsfKernelConfig(
            webbpsf_path=raw[WEBBPSF_PATH_KEY],
            psf_library_path=raw[PSF_LIBRARY_PATH_KEY],
            psf_arcsec=raw[PSF_ARCSEC_KEY],
            normalize=raw.get(NORMALIZE_KEY, NORMALIZE_DEFAULT),
            opd_by_date=raw.get(OPD_BY_DATE_KEY, False),
            correction=correction,
            pixel_box=raw.get(PIXEL_BOX_KEY, False),
            brighter_fatter=kappa,
            joint_stars=joint,
        )
