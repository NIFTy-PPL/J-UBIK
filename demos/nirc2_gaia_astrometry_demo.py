# %% [markdown]
# # NIRC2 narrow camera against Gaia DR3
#
# Checks the sky coordinates `load_nirc2_exposure` gives the pixels of real,
# raw NIRC2 narrow-camera frames: the position angle
# ``ROTPOSN - INSTANGL - beta``, the plate scale, the parity and the
# distortion maps of both published solutions.
#
# For every frame the demo
#
# 1. loads it with `load_nirc2_exposure` and takes the sky position of every
#    pixel centre from the loader's output (``subsample_centers``),
# 2. centroids the bright stars (a Gaussian fit to the core of each),
# 3. matches them to Gaia DR3, propagated to the frame's epoch with the Gaia
#    proper motions (`StarTables`), and
# 4. fits Gaia = s R(theta) loader + shift. theta > 0 means the true position
#    angle of the detector exceeds the loader's, s - 1 the plate-scale error.
#    A flipped sign of beta would give theta = +2 beta (+0.504 deg before
#    2015-04-13, +0.524 deg after), or no match on a crowded field (step 3).
#
# The same stars are then refitted with the loader given other conventions
# (beta added, the maps negated, transposed or swapped, no maps, the other
# solution) through its ``distortion=`` and ``position_angle=`` overrides: the
# right maps and solution have the smallest chi^2. The fitted rotation absorbs
# the sign of beta, which therefore shows in theta alone.
#
# ## Data (downloaded, so this demo is not run in CI)
#
# - Raw NIRC2 narrow-camera frames of a field with Gaia stars, from the Keck
#   Observatory Archive (https://koa.ipac.caltech.edu), full 1024 x 1024
#   frames. Checked on M53 (NGC 5024), Kp, 10 s x 5 coadds: N2.20150402.34994
#   and .35067 (before the realignment), N2.20150505.27289 and .27389 (after),
#   N2.20150505.33068 (after, ROTPOSN - INSTANGL = 90 deg); and on M92, Kp,
#   0.8 s x 60 coadds: N2.20070729.38200 and .28274 (at 0 and 90 deg).
# - The distortion maps ``nirc2_distort_{X,Y}_{pre,post}20150413_v1.fits``
#   from https://github.com/jluastro/nirc2_distortion, in one directory.
#
# ```
# python demos/nirc2_gaia_astrometry_demo.py --maps MAPS_DIR --out OUT_DIR FRAME [FRAME ...]
# ```
#
# The Gaia query is cached in ``OUT_DIR``; the residual plots go there too.

# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from scipy import ndimage, optimize

from jubik.instruments.gaia.star_finder import load_gaia_stars_in_fov
from jubik.instruments.jwst.alignment.star_alignment import StarTables
from jubik.instruments.jwst.throughput.weights import Throughput
from jubik.instruments.keck.nirc2_distortion import (
    DistortionSolution,
    select_distortion,
    service2016,
    yelda2010,
)
from jubik.instruments.keck.nirc2_loader import (
    DetectorNoise,
    FitsImage,
    Nirc2Calibration,
    load_nirc2_exposure,
)
from jubik.wcs import WcsAstropy

# e- per correlated double sample (NIRC2 Observer's Manual); enters the
# centroid weights only
READ_NOISE = 38.0
# added in quadrature to every star: the solutions' quoted residual distortion
# is about 1 mas (Yelda et al. 2010; Service et al. 2016)
FLOOR_MAS = 1.0
BOX = 5  # half-width of the centroid box, about one FWHM at Kp
# CRVAL is the target's position, not the frame's: 15 arcsec off on M53
SEARCH_MAS = 30_000.0
# fewer matched stars: the wrong field or a mirrored frame, whose chance pairs
# number 2 to 3 on M53
MIN_STARS = 6
DEG = 180 / np.pi

# %% [markdown]
# ## The loader's sky coordinates


# %%
def load_offsets(frame: Path, pre, post, **overrides):
    """East and North offsets (mas) of every pixel centre from the frame's
    reference position, as the loader places them; shape (2, ny, nx)."""
    header = fits.getheader(frame)
    centre = SkyCoord(header["CRVAL1"], header["CRVAL2"], unit="deg")
    exposure = load_nirc2_exposure(
        FitsImage(frame),
        calibration=Nirc2Calibration(1 * u.Jy / (u.DN / u.s), "unused, astrometry"),
        throughput=Throughput([1.95, 2.30] * u.um, [1.0, 1.0]),
        psf=np.ones((1, 1)),
        # wide enough to keep the whole detector at any pointing error
        sky_grid=WcsAstropy(centre, shape=300, fov=30 * u.arcsec),
        margin=0,
        detector_noise=DetectorNoise(
            gain=header["GAIN"],
            read_noise=READ_NOISE,
            n_reads=header["MULTISAM"],
            sky=0.0,  # raw frame: the sky is still in it
        ),
        pre_2015=pre,
        post_2015=post,
        bunit="DN",  # raw frames carry no BUNIT; DN summed over coadds
        **overrides,
    )
    full = (0, header["NAXIS2"], 0, header["NAXIS1"])
    assert exposure.diagnostics["bounds"] == full, "not the full frame"
    sky = exposure.target_data.subsample_centers[0]
    east, north = centre.icrs.spherical_offsets_to(sky.icrs)
    offsets = np.stack([east.to_value(u.mas), north.to_value(u.mas)])
    return offsets, exposure


def at(offsets, x, y):
    """The loader's offsets at fractional 0-based pixel positions; bilinear
    between pixel centres, as the loader interpolates the maps."""
    return np.stack(
        [ndimage.map_coordinates(o, [y, x], order=1) for o in offsets], axis=-1
    )


# %% [markdown]
# ## Star centroids


# %%
def gaussian(p, y, x):
    amplitude, x0, y0, sx, sy, background = p
    return (
        amplitude * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))
        + background
    )


def centroids(data, std, threshold=20.0):
    """0-based (x, y) centroids and their 1-sigma errors (pixels) of the
    isolated peaks above ``threshold`` robust sigma, shape (n, 4)."""
    # dead pixels of a raw frame read about 0 against a sky of some 100 DN;
    # with read noise as their only sigma they would dominate the fit
    good = data > 0.3 * np.median(data)
    smooth = ndimage.gaussian_filter(ndimage.median_filter(data, 3), 1.5)
    level = np.median(smooth)
    mad = 1.4826 * np.median(np.abs(smooth - level))
    peaks = (smooth == ndimage.maximum_filter(smooth, 9)) & (
        smooth > level + threshold * mad
    )
    ys, xs = np.nonzero(peaks)
    height = smooth[ys, xs] - level
    ny, nx = data.shape
    found = []
    for x, y, h in zip(xs, ys, height):
        if not (BOX <= x < nx - BOX and BOX <= y < ny - BOX):
            continue
        near = np.hypot(xs - x, ys - y) < 2 * BOX
        # a brighter star nearby (this may be its Airy ring), or a neighbour
        # bright enough to bias the fit
        if np.any(near & (height > h)) or np.any(
            near & (height > 0.1 * h) & (height < h)
        ):
            continue
        cut = np.s_[y - BOX : y + BOX + 1, x - BOX : x + BOX + 1]
        yy, xx = np.mgrid[cut]
        d, s, g = data[cut], std[cut], good[cut]
        start = [d.max() - np.median(d), x, y, 2.0, 2.0, np.median(d)]
        fit = optimize.least_squares(
            lambda p: ((gaussian(p, yy, xx) - d) / s)[g], start
        )
        chi2 = 2 * fit.cost / (g.sum() - len(start))
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * max(chi2, 1.0)
        found.append([*fit.x[1:3], *np.sqrt(np.diag(cov)[1:3])])
    return np.array(found).reshape(-1, 4)


# %% [markdown]
# ## Gaia DR3 at the frame's epoch


# %%
def gaia(offsets, centre, time, cache):
    """Gaia stars with a proper motion, out to SEARCH_MAS past the frame:
    offsets (mas) at ``time`` from ``centre`` and their (2, 2) covariances
    (mas^2)."""
    corners = offsets[:, [0, 0, -1, -1], [0, -1, -1, 0]]
    corners = centre.spherical_offsets_by(
        *((corners + np.sign(corners) * SEARCH_MAS) * u.mas)
    )
    table = load_gaia_stars_in_fov(corners, library_path=str(cache))
    xy, cov = [], []
    for star in StarTables([table], [time]).get_stars():
        p = star.position_at(time)
        xy.append(
            [
                o.to_value(u.mas)
                for o in centre.spherical_offsets_to(SkyCoord(p.ra, p.dec))
            ]
        )
        jacobian = star.epoch_jacobian(time)
        cov.append(jacobian @ star.astrometric_covariance @ jacobian.T)
    return np.array(xy), np.array(cov)


# %% [markdown]
# ## Matching and fitting


# %%
def design(xy, affine):
    """Rows (2n, k) of Gaia = model(xy) for the similarity (a, b, e, n),
    E' = a E + b N + e, N' = -b E + a N + n, or the affine transform."""
    e, n = xy.T
    one, zero = np.ones_like(e), np.zeros_like(e)
    if affine:
        rows_e = [e, n, one, zero, zero, zero]
        rows_n = [zero, zero, zero, e, n, one]
    else:
        rows_e = [e, n, one, zero]
        rows_n = [n, -e, zero, one]
    return np.stack([np.stack(rows_e, -1), np.stack(rows_n, -1)], axis=1)


def fit(loader, target, cov, affine=False):
    """Generalized least squares; returns parameters, their covariance scaled
    by max(chi^2_red, 1), residuals (n, 2) in mas, chi^2_red."""
    a = design(loader, affine)  # (n, 2, k)
    w = np.linalg.inv(cov)
    normal = np.einsum("nij,njk,nkl->il", a.transpose(0, 2, 1), w, a)
    rhs = np.einsum("nij,njk,nk->i", a.transpose(0, 2, 1), w, target)
    p = np.linalg.solve(normal, rhs)
    r = target - np.einsum("nij,j->ni", a, p)
    chi2 = np.einsum("ni,nij,nj->", r, w, r) / (2 * len(r) - len(p))
    return p, np.linalg.inv(normal) * max(chi2, 1.0), r, chi2


def rotation_scale(p, pcov):
    """theta (deg) and s of a similarity fit, with their 1-sigma errors."""
    a, b = p[:2]
    s = np.hypot(a, b)
    jac = np.array([[-b, a], [a, b]]) / np.array([[s**2], [s]])
    err = np.sqrt(np.diag(jac @ pcov[:2, :2] @ jac.T))
    return np.arctan2(b, a) * DEG, err[0] * DEG, s, err[1]


def match(detected, catalogue, tolerance=30.0):
    """Index pairs (detection, Gaia): the most voted shift first, then nearest
    neighbours after a similarity fit, within ``tolerance`` mas. Raises
    ValueError for fewer than MIN_STARS pairs: the wrong field, a mirrored
    frame, or a position angle off by more than about 0.3 deg."""
    # shortcut: votes a shift only, so a 0.5 deg position-angle error raises on
    # M92 at 90 deg; vote over trial rotations (tie-broken towards 0) if needed
    shifts = (catalogue[None] - detected[:, None]).reshape(-1, 2)
    edges = np.arange(-SEARCH_MAS, SEARCH_MAS + 1, 100.0)
    votes, *_ = np.histogram2d(*shifts.T, bins=[edges, edges])
    i, j = np.unravel_index(votes.argmax(), votes.shape)
    moved = detected + [edges[i] + 50, edges[j] + 50]
    # wide first, for the voted shift; then refitted within ``tolerance``, so
    # a wrong pair from the wide search cannot pull a true one out
    for radius in (3 * tolerance, tolerance, tolerance):
        distance = np.linalg.norm(catalogue[None] - moved[:, None], axis=-1)
        # mutual nearest neighbours
        pairs = [
            (k, m)
            for m, k in enumerate(distance.argmin(0))
            if distance[k, m] < radius and distance[k].argmin() == m
        ]
        if len(pairs) < MIN_STARS:
            raise ValueError(f"only {len(pairs)} stars match Gaia")
        k, m = np.array(pairs).T
        p, *_ = fit(detected[k], catalogue[m], np.tile(np.eye(2), (len(k), 1, 1)))
        moved = np.einsum("nij,j->ni", design(detected, False), p)
    close = np.linalg.norm(moved[k] - catalogue[m], axis=-1) < tolerance
    if close.sum() < MIN_STARS:
        raise ValueError(f"only {close.sum()} stars match Gaia")
    return k[close], m[close]


# %% [markdown]
# ## One frame


# %%
def check(frame: Path, pre, post, out: Path):
    header = fits.getheader(frame)
    time = Time(header["MJD-OBS"], format="mjd", scale="utc")
    solution = select_distortion(header["DATE-OBS"], pre, post)
    other = post if solution is pre else pre
    nominal = header["ROTPOSN"] - header["INSTANGL"]

    offsets, exposure = load_offsets(frame, pre, post)
    data = exposure.target_data.data[0]
    std = exposure.target_data.std[0]
    found = centroids(data, std)
    mas_per_pixel = solution.pixel_scale.to_value(u.mas)
    detected = at(offsets, found[:, 0], found[:, 1])
    centre = SkyCoord(header["CRVAL1"], header["CRVAL2"], unit="deg").icrs
    catalogue, gaia_cov = gaia(offsets, centre, time, out)

    k, m = match(detected, catalogue)
    centroid_cov = np.einsum(
        "ni,ij->nij", (found[k, 2:4] * mas_per_pixel) ** 2, np.eye(2)
    )
    cov = gaia_cov[m] + centroid_cov + FLOOR_MAS**2 * np.eye(2)

    def refit(**overrides):
        moved = at(load_offsets(frame, pre, post, **overrides)[0], *found[k, :2].T)
        p, pcov, r, chi2 = fit(moved, catalogue[m], cov)
        return (*rotation_scale(p, pcov), np.sqrt(np.mean(r**2, 0)), chi2, p, r, moved)

    def solution_like(name, dx, dy):
        s = solution
        return DistortionSolution(name, dx, dy, s.pixel_scale, s.angle_offset)

    beta = solution.angle_offset
    variants = {
        "loader (date-chosen solution)": {},
        "beta added": dict(position_angle=nominal * u.deg + beta),
        "maps negated": dict(
            distortion=solution_like("negated", -solution.dx, -solution.dy)
        ),
        "maps transposed": dict(
            distortion=solution_like("transposed", solution.dx.T, solution.dy.T)
        ),
        "maps swapped (dx <-> dy)": dict(
            distortion=solution_like("swapped", solution.dy, solution.dx)
        ),
        "no maps": dict(
            distortion=solution_like(
                "none", np.zeros_like(solution.dx), np.zeros_like(solution.dy)
            )
        ),
        f"other solution ({other.name.split(' (')[0]})": dict(distortion=other),
    }
    results = {name: refit(**kw) for name, kw in variants.items()}

    theta, theta_err, s, s_err, rms, chi2, p, r, moved = results[
        "loader (date-chosen solution)"
    ]
    pa, pa_cov, *_ = fit(moved, catalogue[m], cov, affine=True)
    det = pa[0] * pa[4] - pa[1] * pa[3]
    print(
        f"\n## {frame.name}: DATE-OBS {header['DATE-OBS']} {header.get('UTC', '')}, "
        f"{solution.name.split(' (')[0]}, ROTPOSN - INSTANGL = {nominal:.3f} deg, "
        f"loader PA {exposure.diagnostics['position_angle']:.3f}\n"
        f"{len(found)} centroids, {len(catalogue)} Gaia stars with proper motions, "
        f"{len(k)} matched; Gaia epoch offset "
        f"{(time - Time('J2016.0')).to_value(u.yr):+.2f} yr\n"
        f"affine fit: determinant {det:+.5f}, shear and differential scale "
        f"{(pa[1] + pa[3]) / 2:+.2e} and {(pa[0] - pa[4]) / 2:+.2e} "
        f"(+-{np.sqrt(pa_cov[1, 1] + pa_cov[3, 3]) / 2:.1e})\n"
    )
    print("| loader input | theta (deg) | s - 1 (%) | rms E, N (mas) | chi2_red |")
    print("|---|---|---|---|---|")
    for name, (t, te, sc, se, rr, c2, *_) in results.items():
        print(
            f"| {name} | {t:+.4f} +- {te:.4f} | {100 * (sc - 1):+.3f} +- "
            f"{100 * se:.3f} | {rr[0]:.2f}, {rr[1]:.2f} | {c2:.2f} |"
        )

    figure, axis = plt.subplots(figsize=(5, 5))
    arrow = 200.0  # sky mas per residual mas
    axis.quiver(*moved.T, *r.T, angles="xy", scale_units="xy", scale=1 / arrow)
    axis.quiver(
        [-4500], [-4500], [5 * arrow], [0], angles="xy", scale_units="xy", scale=1
    )
    axis.annotate("5 mas", (-4500, -4300))
    axis.set(
        xlabel="East offset (mas)",
        ylabel="North offset (mas)",
        title=f"{frame.stem}: Gaia - fit",
        aspect="equal",
    )
    axis.invert_xaxis()
    figure.savefig(out / f"{frame.stem}_residuals.png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return results


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="+", type=Path, help="raw NIRC2 frames")
    parser.add_argument("--maps", type=Path, required=True, help="distortion maps")
    parser.add_argument("--out", type=Path, required=True, help="cache and plots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pre = yelda2010(
        args.maps / "nirc2_distort_X_pre20150413_v1.fits",
        args.maps / "nirc2_distort_Y_pre20150413_v1.fits",
    )
    post = service2016(
        args.maps / "nirc2_distort_X_post20150413_v1.fits",
        args.maps / "nirc2_distort_Y_post20150413_v1.fits",
    )
    for frame in args.frames:
        check(frame, pre, post, args.out)
