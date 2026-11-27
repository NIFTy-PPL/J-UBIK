"""``psf.joint_stars`` with ``gaia_alignment``: every field star is one exact
model under the learned correction and kappa; a star at a Gaia position is
placed there, moved by its exposure's shift and its astrometry."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jubik.instruments.jwst import jwst_likelihoods
from jubik.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.psf_correction import (
    PsfCorrection,
    StarStamp,
    _StarModel,
    load_star_stamps,
    save_star_stamps,
    with_pixel_box,
)

from test_brighter_fatter import BAND, DN_PER_UNIT, KAPPA, bfe, latents
from test_joint_psf_stars import (
    SUB,
    width_at_fluence,
    DeterministicJwstData,
    base_kernel,
    energy,
    fit,
    joint_psf,
    kappa_of,
    texture_and_width,
    tree,
)
from test_jwst_likelihoods import (
    JWST_DATA_SITES,
    SKY_DOMAIN,
    make_config,
    make_grid,
)
from test_jwst_psf_broadening import wide_build_webb_psf
from test_psf_correction import SIZE

H = SIZE // 2
FILES = ("fake_0.fits", "fake_1.fits")
DATES = ("2023-01-01T00:00:00.000", "2023-06-01T00:00:00.000")
SHIFT_STD = 0.1  # arcsec, make_config's shift prior
GRID_PIXEL = 0.1  # arcsec, make_grid's


# the tests' Gaia stars on the fake detector, (x, y): fake_gaia's two, the
# third of `add_star`'s tests and the edge star
STARS = ((31.5, 31.5), (14.42, 14.36), (45.3, 20.6), (0.3, 30.4))


class EpochJwstData(DeterministicJwstData):
    """`DeterministicJwstData` with fake_1 five months after fake_0 and a blob
    of light at each of `STARS`, so a Gaia cutout holds a star."""

    def __init__(self, filepath):
        super().__init__(filepath)
        self.dm.meta.observation.date_beg = DATES[int(Path(filepath).stem[-1])]
        y, x = np.mgrid[: self.shape[0], : self.shape[1]]
        for x0, y0 in STARS:
            r2 = (x - x0) ** 2 + (y - y0) ** 2
            self.dm.data = self.dm.data + 50 * np.exp(-0.5 * r2 / 1.5**2)


@pytest.fixture
def gaia(monkeypatch, fake_gaia):
    """The `fake_gaia` stars with 100 mas position and 10 mas/yr proper-motion
    errors, so their astrometry latents move them by pixels; the files'
    `stars_data` lands in the returned list at each build."""
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, EpochJwstData)
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", wide_build_webb_psf)
    _, table = fake_gaia
    table["ra_error"], table["dec_error"] = [100.0, 100.0] * u.mas, [
        80.0,
        120.0,
    ] * u.mas
    table["pmra_error"] = table["pmdec_error"] = [10.0, 10.0] * u.mas / u.yr
    loaded, load_data = [], jwst_likelihoods.load_data

    def spy(*args, **kwargs):
        results = load_data(*args, **kwargs)
        loaded.append(results.stars_data)
        return results

    monkeypatch.setattr(jwst_likelihoods, "load_data", spy)
    return table, loaded


def exposure_wcs():
    return EpochJwstData("fake_0.fits").wcs


def catalog(table, i):
    return SkyCoord(table["ra"][i], table["dec"][i], unit="deg")


# stamp i: (file, centre pixel (x, y) in it): Gaia stars 1 and 2 (or no star
# without a second row) in both files, as A, B, B, A, so the stars' frames
# interleave; a stamp at no Gaia star; star 1 in a file outside the fit
def stamp_sites(table):
    def centre(i):  # x != y, so a swap shows
        xy = exposure_wcs().world_to_pixel(catalog(table, i))
        return np.round(np.array(xy, float) + [0.2, -0.3])

    a, b = centre(0), centre(1) if len(table) > 1 else np.array([50.0, 10.0])
    return [  # star 1's fake_1 stamp first: each stamp ties in its own file
        ("fake_1.fits", *a),
        ("fake_0.fits", *b),
        ("fake_1.fits", *b),
        ("fake_0.fits", *a),
        ("fake_0.fits", 10.0, 50.0),
        ("other.fits", *a),
    ]


def add_star(table, source_id, x, y, row=None):
    """A Gaia star at detector pixel (x, y), with the fixture's errors."""
    sky = exposure_wcs().pixel_to_world(x, y)
    values = (source_id, sky.ra.deg, sky.dec.deg, 0.0, 0.0, 100.0, 100.0, 10.0, 10.0)
    table.insert_row(len(table) if row is None else row, values + (0.0,) * 6)


def write_stamps(path, table, data=None, seed=0):
    """The stamps of `stamp_sites` with `data` (default: noise around a bump),
    each with its file's DN per data unit, as the star-fit script writes them."""
    rng = np.random.default_rng(seed)
    sites = stamp_sites(table)
    yx = np.mgrid[:SIZE, :SIZE] - H
    if data is None:
        bump = 50 * np.exp(-0.5 * (yx**2).sum(0) / 2.0)
        data = [bump + 5 + rng.normal(size=(SIZE, SIZE)) for _ in sites]
    std = [np.full((SIZE, SIZE), 0.5 + 0.1 * i) for i in range(len(sites))]
    mask = np.ones((SIZE, SIZE), bool)
    mask[0, :3] = False
    dn = [dict(zip(FILES, DN_PER_UNIT)).get(site[0], 250.0) for site in sites]
    stamps = [
        StarStamp(d, s, mask, base_kernel(), 1e4 * (i + 1), u_)
        for i, (d, s, u_) in enumerate(zip(data, std, dn))
    ]
    file, x, y = (np.array(c) for c in zip(*sites))
    sky = exposure_wcs().pixel_to_world(x, y)
    save_star_stamps(
        path, stamps, SUB, exposure=file, x=x, y=y, ra=sky.ra.deg, dec=sky.dec.deg
    )
    return stamps


def warned(caplog, text):
    return any(
        r.levelname == "WARNING" and text in r.getMessage() for r in caplog.records
    )


def build(tmp_path, psf, fov="0.5arcsec", **telescope):
    tmp_path.mkdir(exist_ok=True)
    cfg = make_config(tmp_path, n_files=2, gaia=True, fov=fov)
    cfg["telescope"]["psf"] |= dict(psf)
    cfg["telescope"] |= telescope
    return build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


# --- an independent model of one star -----------------------------------------------


def true_position(table, i, x, e, wcs, corner):
    """Star `i`'s position in exposure `e`'s stamp at `corner` (detector (x, y)
    of its pixel 0), subsampled (x, y): its Gaia astrometry moved by
    ``star_<id>_astrometry`` (diagonal covariance), propagated on the sphere to
    the epoch, then moved on the sky grid against the exposure's shift."""
    star = table[i]
    v = np.asarray(x[f"star_{star['source_id']}_astrometry"])
    error = [star[f"{n}_error"] for n in ("ra", "dec", "pmra", "pmdec")]
    d = [float(err) * float(z) for err, z in zip(error, v)]  # mas, mas/yr
    dec = star["dec"] * u.deg + d[1] * u.mas
    astrometry = SkyCoord(
        ra=star["ra"] * u.deg + d[0] * u.mas / np.cos(star["dec"] * u.deg),
        dec=dec,
        pm_ra_cosdec=(star["pmra"] + d[2]) * u.mas / u.yr,
        pm_dec=(star["pmdec"] + d[3]) * u.mas / u.yr,
        obstime=Time("J2016.0"),
    )
    position = astrometry.apply_space_motion(Time(DATES[e], scale="utc"))
    grid = make_grid().spatial
    row, column = grid.world_to_indices_yx(position)
    shift = SHIFT_STD * np.asarray(x[f"{BAND}_shift"])[e] / GRID_PIXEL
    moved = grid.indices_yx_to_world(row - shift[0], column - shift[1])
    xy = np.array(wcs.world_to_pixel(moved), float) - corner
    return (xy + 0.5) * SUB - 0.5


def render(stamp, texture, width, kappa, xy, flux, background, gradient, n=512, o=128):
    """`stamp`'s model with numpy alone: the kernel times exp(texture) at its
    sum, a Gaussian and the sub-pixel box on a canvas of `n` subsampled pixels
    with the stamp's pixel 0 at `o`, its centre moved to `xy` (subsampled (x, y)
    of the stamp) by a Fourier phase, binned, plus background and gradient, and
    its charge moved at `kappa` per DN/int.

    The phase interpolates periodically on the canvas: the model's canvas, 12
    detector pixels beyond the stamp, and a wide one differ by ~1e-6."""
    k = np.asarray(stamp.kernel, float)
    t = np.pad(texture, (k.shape[0] - texture.shape[0]) // 2)
    k = k * np.exp(t) * k.sum() / (k * np.exp(t)).sum()
    size = stamp.data.shape[0]
    canvas = np.zeros((n, n))
    r = k.shape[0] // 2
    canvas[n // 2 - r : n // 2 + r + 1, n // 2 - r : n // 2 + r + 1] = k
    f = np.fft.fftfreq(n)
    fy, fx = f[:, None], f[None, :]
    sy, sx = o + xy[1] - n // 2, o + xy[0] - n // 2
    transfer = np.exp(-2 * np.pi**2 * width**2 * (fy**2 + fx**2))
    transfer = transfer * np.sinc(fy) * np.sinc(fx)
    transfer = transfer * np.exp(-2j * np.pi * (fy * sy + fx * sx))
    over = np.fft.ifft2(np.fft.fft2(canvas) * transfer).real
    over = over[o : o + size * SUB, o : o + size * SUB]
    star = over.reshape(size, SUB, size, SUB).sum((1, 3))
    yx = np.mgrid[:size, :size] - size // 2
    model = flux * star + background + gradient[0] * yx[0] + gradient[1] * yx[1]
    if kappa is None:
        return model
    dn = stamp.dn_per_unit
    return np.asarray(bfe(jnp.asarray(model * dn), (kappa,))) / dn


def chi2(stamps, models):
    return sum(
        0.5 * float(np.sum(((s.data - m) / s.std)[s.mask] ** 2))
        for s, m in zip(stamps, models)
    )


def standalone(
    stamps, x, prefix, positions, kappa=True, canvas=(512, 128), widths=None
):
    """The energy of `stamps` with each star at `positions` (subsampled (x, y),
    None for a free star at its latent), everything else from `x`; `canvas` is
    `render`'s ``(n, o)``, `widths` each stamp's (default: the one width)."""
    start = _StarModel(stamps, SUB)  # the data's starting flux and background
    texture, width = texture_and_width(x)
    widths = [width] * len(stamps) if widths is None else widths
    k = float(kappa_of(x)) if kappa else None
    free = (
        iter(np.asarray(x[f"{prefix}star_position"]))
        if any(p is None for p in positions)
        else ()
    )
    models = []
    for i, (stamp, xy, width) in enumerate(zip(stamps, positions, widths)):
        if xy is None:
            dy, dx = 0.5 * next(free)
            xy = (H + np.array([dx, dy]) + 0.5) * SUB - 0.5
        flux = start.flux0[i] * np.exp(x[f"{prefix}star_flux"][i])
        background = (
            start.background0[i]
            + 10 * start.noise[i] * x[f"{prefix}star_background"][i]
        )
        gradient = start.noise[i] * np.asarray(x[f"{prefix}star_gradient"][i])
        models.append(
            render(stamp, texture, width, k, xy, flux, background, gradient, *canvas)
        )
    return chi2(stamps, models)


def gaia_cutouts(table, x, stars_data, frames_of):
    """The Gaia cutouts of ``frames_of``, {table row: exposures}, as stamps,
    and their stars' true positions in them."""
    stamps, positions = [], []
    for i, exposures in frames_of.items():
        frames = stars_data[table["source_id"][i]]
        for k, e in enumerate(frames.observation_ids):
            if e in exposures:
                data, std = frames.data[k], frames.std[k]
                stamps.append(
                    StarStamp(
                        data, std, frames.mask[k], frames.psf[k], 0.0, DN_PER_UNIT[e]
                    )
                )
                inside = (frames.star_in_subsampled_pixels[k] + 0.5) / SUB - 0.5
                corner = np.round(frames.star_in_detector_pixels[k] - inside)
                positions.append(true_position(table, i, x, e, exposure_wcs(), corner))
    return stamps, positions


def test_every_star_is_one_exact_model(gaia, tmp_path, caplog):
    """The stamps at Gaia stars 1 and 2 sit at their catalog positions moved by
    the shift and their astrometry, the other stamps at their own latents, and
    star 3, without a stamp, is its Gaia cutouts at Gaia's position: each
    term's energy is that of an independent model of its stars."""
    table, loaded = gaia
    add_star(table, 3, 45.3, 20.6)
    stamps = write_stamps(tmp_path / "s.npz", table)
    products = build(tmp_path / "b", joint_psf(tmp_path / "s.npz"))
    assert products.alignment is None  # no bilinear Gaia term: no star twice
    tied, cutouts = products.target.joint_stars
    x = latents(products)
    domain = tree(tied.domain) | tree(cutouts.domain)
    # four Gaia frames are stamps, two cutouts; two stamps are free
    assert domain[f"{BAND}_star_position"].shape == (2, 2)
    assert domain[f"{BAND}_gaia_star_flux"].shape == (2,)
    assert {"star_1_astrometry", "star_2_astrometry"} <= set(tree(tied.domain))
    assert {"star_3_astrometry", f"{BAND}_shift"} <= set(tree(cutouts.domain))
    assert warned(caplog, "stamps of ['other.fits'], no file of the fit, keep free")

    wcs, sites = exposure_wcs(), stamp_sites(table)
    corner = [np.array(s[1:]) - H for s in sites]
    positions = [
        true_position(table, i, x, e, wcs, corner[j])
        for j, (i, e) in enumerate([(0, 1), (1, 0), (1, 1), (0, 0)])
    ]
    pad = SUB * 12  # the model's canvas, so only the positions' linearization differs
    canvas = (SUB * SIZE + 2 * pad, pad)
    want = standalone(stamps, x, f"{BAND}_", positions + [None, None], canvas=canvas)
    assert np.isclose(energy(tied, x), want, rtol=2e-8)  # 5e-9

    (stars_data,) = loaded
    gaia_stamps, positions = gaia_cutouts(table, x, stars_data, {2: [0, 1]})
    want = standalone(gaia_stamps, x, f"{BAND}_gaia_", positions)
    assert np.isclose(energy(cutouts, x), want, rtol=3e-6)  # 1e-6: the wide canvas


def test_the_gaia_cutouts_take_the_width_law_at_their_peak(gaia, tmp_path):
    """With a fluence, a Gaia cutout's width is the law's at its peak above
    the cutout's edge, in DN/int."""
    table, loaded = gaia
    add_star(table, 3, 45.3, 20.6)
    write_stamps(tmp_path / "s.npz", table)
    # a fluence about the fake cutouts' peaks (~1e3 DN/int), the width there wider
    products = build(tmp_path / "b", joint_psf(tmp_path / "s.npz", fluence=1e3))
    _, cutouts = products.target.joint_stars
    x = latents(products)
    x[f"{BAND}_psf_width_at_fluence"] = x[f"{BAND}_psf_width"] + 1.0
    w0, w1 = texture_and_width(x)[1], width_at_fluence(x)
    (stars_data,) = loaded
    stamps, positions = gaia_cutouts(table, x, stars_data, {2: [0, 1]})
    widths = []
    for stamp in stamps:
        edge = np.ones(stamp.data.shape, bool)
        edge[1:-1, 1:-1] = False
        peak = stamp.data[stamp.mask].max() - np.median(stamp.data[stamp.mask & edge])
        ratio = peak * stamp.dn_per_unit / 1e3
        widths.append(np.sqrt(max(w0**2 + (w1**2 - w0**2) * ratio, 1e-12)))
    assert min(widths) > 1.1 * w0  # the law widens them
    want = standalone(stamps, x, f"{BAND}_gaia_", positions, widths=widths)
    assert np.isclose(energy(cutouts, x), want, rtol=3e-6)


def test_the_nearest_gaia_star_takes_a_stamp(gaia, tmp_path):
    """A Gaia star 0.14 arcsec from star 1, listed before it, does not take
    star 1's stamps: they go to star 1, the neighbour gets its cutouts."""
    table, _ = gaia
    write_stamps(tmp_path / "s.npz", table)
    x, y = exposure_wcs().world_to_pixel(catalog(table, 0))
    add_star(table, 9, x + 1.6, y - 1.5, row=0)
    products = build(tmp_path / "b", joint_psf(tmp_path / "s.npz"))
    tied, cutouts = products.target.joint_stars
    assert "star_1_astrometry" in tree(tied.domain)
    assert "star_9_astrometry" not in tree(tied.domain)
    assert "star_1_astrometry" not in tree(cutouts.domain)
    assert tree(cutouts.domain)[f"{BAND}_gaia_star_flux"].shape == (2,)


def test_a_cutout_masked_whole_is_left_out(gaia, tmp_path):
    """A corner mask over star 3, whose cutouts are then all masked: no term
    for them (the bilinear term would have kept empty ones)."""
    table, _ = gaia
    add_star(table, 3, 45.3, 20.6)
    write_stamps(tmp_path / "s.npz", table)
    c = catalog(table, 2)
    corners = [
        c.spherical_offsets_by(dx * u.arcsec, dy * u.arcsec)
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    mask = {
        "ra": [f"{p.ra.deg}deg" for p in corners],
        "dec": [f"{p.dec.deg}deg" for p in corners],
    }
    psf = joint_psf(tmp_path / "s.npz")
    assert len(build(tmp_path / "a", psf).target.joint_stars) == 2
    products = build(tmp_path / "b", psf, corner_mask_1=mask)
    (tied,) = products.target.joint_stars
    assert "star_3_astrometry" not in tree(tied.domain)


def test_the_gradient_is_the_sum_over_the_terms(gaia, tmp_path):
    """The texture's, width's, kappa's and the shifts' gradients of the whole
    likelihood are the lens's plus each star term's, and every term has one."""
    table, _ = gaia
    add_star(table, 3, 45.3, 20.6)
    write_stamps(tmp_path / "s.npz", table)
    products = build(tmp_path / "b", joint_psf(tmp_path / "s.npz"))
    x = latents(products)
    lens = replace(products.target, joint_stars=[]).likelihood
    terms = [lens, *products.target.joint_stars]
    assert len(terms) == 3

    def grad(likelihood):
        return jax.grad(lambda y: likelihood(jft.Vector(y)))(x)

    total, parts = grad(products.target.likelihood), [grad(t) for t in terms]
    keys = [f"{BAND}_psf_texturexi", f"{BAND}_psf_width"]
    keys += [f"{BAND}_psf_brighter_fatter", f"{BAND}_shift"]
    for key in keys:
        each = [np.asarray(p[key]) for p in parts]
        assert all(np.any(g != 0) for g in each), key
        np.testing.assert_allclose(total[key], sum(each), rtol=1e-10, err_msg=key)


def test_off_keeps_the_bilinear_gaia_term(gaia, tmp_path):
    """Without joint_stars, gaia_alignment stays the bilinear alignment term:
    its energy and gradient at fixed latents are those of the code before the
    joint Gaia stars (5d97a15d, the same test), the astrometry fitted."""
    products = build(tmp_path, {})
    assert products.target.joint_stars == []
    alignment = products.alignment.likelihood.likelihood
    assert "star_1_astrometry" in alignment.domain.tree
    x = jft.random_like(jax.random.PRNGKey(4), alignment.domain)
    grad = jax.grad(alignment)(x)
    norm = np.sqrt(sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grad)))
    assert np.isclose(float(alignment(x)), ENERGY_BILINEAR, rtol=1e-13)
    assert np.isclose(norm, GRADIENT_BILINEAR, rtol=1e-12)


# the values at 5d97a15d
ENERGY_BILINEAR, GRADIENT_BILINEAR = 35023.31715750818, 4.084489153617946


def test_a_star_with_parallax_is_flagged(gaia, tmp_path, caplog):
    table, _ = gaia
    table["parallax"] = [0.3, 2.0] * u.mas
    write_stamps(tmp_path / "s.npz", table)
    build(tmp_path / "b", joint_psf(tmp_path / "s.npz"))
    assert warned(caplog, "Gaia stars [2] have a parallax of 1 mas or more")


def test_stamps_without_their_place_raise(gaia, tmp_path):
    table, _ = gaia
    stamps = write_stamps(tmp_path / "s.npz", table)
    save_star_stamps(
        tmp_path / "bare.npz", stamps, SUB, exposure=["fake_0.fits"] * len(stamps)
    )
    with pytest.raises(
        ValueError, match=r"needs the stamps' \['dec', 'ra', 'x', 'y'\]"
    ):
        build(tmp_path / "b", joint_psf(tmp_path / "bare.npz"))


def test_the_joint_fit_recovers_a_known_shift_and_correction(gaia, tmp_path):
    """Lens data made with a correction file's kernel at known shifts, 3% noise,
    and stamps made with the same correction and shifts, two of them of Gaia
    star 1: the joint fit, the sky held at its truth, recovers the kernel, the
    width and both exposures' shifts; the stars alone recover the shifts too,
    as Gaia pins them. The stars' data come from the model under test: this
    checks the fit, `test_every_star_is_one_exact_model` the placement."""
    table, _ = gaia
    table.remove_row(1)  # its cutouts would hold noise alone
    table["ra_error"], table["dec_error"] = [0.1] * u.mas, [0.2] * u.mas  # Gaia's
    table["pmra_error"], table["pmdec_error"] = [0.1] * u.mas / u.yr, [
        0.3
    ] * u.mas / u.yr
    stamps = write_stamps(tmp_path / "s.npz", table)
    psf = joint_psf(tmp_path / "s.npz", kappa=None, orthogonal=True)
    products = build(tmp_path / "a", psf)
    truth = latents(products, seed=11)
    truth[f"{BAND}_psf_width"] = np.array(1.0)
    shift = np.array([[0.4, -0.3], [-0.2, 0.5]])  # 40, -30, -20, 50 mas
    truth[f"{BAND}_shift"] = shift
    truth["star_1_astrometry"] = np.zeros(4)  # the fit's prior holds it there
    texture, width = texture_and_width(truth, orthogonal=True)

    (stars,) = products.target.joint_stars
    data, std = np.array([s.data for s in stamps]), np.array([s.std for s in stamps])
    mask = np.array([s.mask for s in stamps])
    residual = np.asarray(stars.normalized_residual(jft.Vector(truth)))
    noise = np.random.default_rng(3).normal(size=residual.shape)
    data[mask] = data[mask] + std[mask] * (noise - residual)  # model + noise
    stamps = write_stamps(tmp_path / "s.npz", table, data=list(data))
    products = build(tmp_path / "b", psf)

    correction = PsfCorrection(BAND, SUB, texture, width, pixel_box=True)
    correction.save(tmp_path / "c.npz")
    fixed = build(
        tmp_path / "c",
        {"pixel_box": True, "correction": {BAND: str(tmp_path / "c.npz")}},
    )
    sky = products.target.sky_filter(truth)
    image = np.asarray(fixed.target.likelihoods[0].response(truth | sky))
    (single,) = products.target.likelihoods
    lens_noise = 3e-2 * np.abs(image).max()
    lens = single.builder.data.copy()
    lens[single.builder.mask] = image + lens_noise * np.random.default_rng(2).normal(
        size=image.shape
    )
    single.builder.data, single.builder.std = lens, np.full(lens.shape, lens_noise)

    def fit_with_the_sky_held(likelihood):
        domain = {k: v for k, v in tree(likelihood.domain).items() if k != "sky"}
        held = jft.Model(
            lambda x: jft.Vector(tree(x) | {"sky": truth["sky"]}), domain=domain
        )
        return fit(likelihood.amend(held, domain=jft.Vector(domain)))

    true_kernel = with_pixel_box(correction.apply(base_kernel()))
    untouched = np.abs(with_pixel_box(base_kernel()) - true_kernel).sum()

    def shift_error(learned):  # mas
        return 1e3 * SHIFT_STD * np.abs(learned[f"{BAND}_shift"] - shift).max()

    learned = fit_with_the_sky_held(products.target.likelihood)
    kernel = np.asarray(single.response.psf.kernel(learned))
    kernel_error = np.abs(kernel - true_kernel).sum() / untouched
    width_error = abs(texture_and_width(learned, orthogonal=True)[1] / width - 1)
    assert kernel_error < 0.1 and width_error < 0.02
    # over noise seeds: the lens alone 1-2 mas, the joint fit 1-1.4, the stars
    # alone 0.5-2.1; the shifts are 20 to 50 mas
    assert shift_error(learned) < 2.5

    (stars,) = products.target.joint_stars
    alone = fit(stars)
    assert shift_error(alone) < 3.0


def winged_build_webb_psf(
    camera, filter, center_pixel, webbpsf_path, subsample, *a, **k
):
    """A core with strong wings 30 detector pixels out (at subsample 2)."""
    r = np.hypot(*np.mgrid[-60:61, -60:61]) / subsample
    psf = np.exp(-0.5 * r**2) + 0.05 / (1 + (r / 3) ** 2)
    return psf / psf.sum()


def test_a_star_far_off_its_cutout_centre_does_not_wrap(gaia, monkeypatch, tmp_path):
    """A Gaia star at the detector's edge, 15 pixels off the middle of its
    cutout, whose kernel's wings would wrap around a canvas 12 pixels wider than
    the cutout into it: its energy is the independent model's."""
    table, loaded = gaia
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", winged_build_webb_psf)
    add_star(table, 3, 0.3, 30.4)
    write_stamps(tmp_path / "s.npz", table)
    products = build(tmp_path / "b", joint_psf(tmp_path / "s.npz"), fov="2arcsec")
    _, cutouts = products.target.joint_stars
    x = latents(products)
    (stars_data,) = loaded
    frames = stars_data[3]
    inside = (frames.star_in_subsampled_pixels + 0.5) / SUB - 0.5
    far = inside - frames.data.shape[-1] // 2
    assert np.abs(far).max() > 12
    stamps, positions = gaia_cutouts(table, x, stars_data, {2: [0, 1]})
    want = standalone(stamps, x, f"{BAND}_gaia_", positions)
    assert np.isclose(energy(cutouts, x), want, rtol=3e-6)


def test_a_second_stamp_of_a_gaia_star_stays_free(gaia, tmp_path):
    """Two stamps of star 1 in one file: the nearer sits at Gaia, the other
    keeps its free position, so the frame is placed once."""
    table, _ = gaia
    stamps = write_stamps(tmp_path / "s.npz", table)
    _, _, extra = load_star_stamps(tmp_path / "s.npz")
    twin = {k: np.append(v, v[:1]) for k, v in extra.items()}
    twin["x"][-1] += 1  # one pixel off: farther from star 1
    sky = exposure_wcs().pixel_to_world(twin["x"][-1], twin["y"][-1])
    twin["ra"][-1], twin["dec"][-1] = sky.ra.deg, sky.dec.deg
    save_star_stamps(tmp_path / "t.npz", stamps + stamps[:1], SUB, **twin)
    products = build(tmp_path / "b", joint_psf(tmp_path / "t.npz"))
    (tied,) = products.target.joint_stars
    assert tree(tied.domain)[f"{BAND}_star_position"].shape == (3, 2)


def test_no_gaia_star_in_the_files_is_logged(gaia, tmp_path, caplog):
    table, _ = gaia
    write_stamps(tmp_path / "s.npz", table)
    table.remove_rows(slice(None))
    products = build(tmp_path / "b", joint_psf(tmp_path / "s.npz"))
    (tied,) = products.target.joint_stars
    assert tree(tied.domain)[f"{BAND}_star_position"].shape == (6, 2)
    assert warned(caplog, "gaia_alignment finds no star in the fit's files")


def box_around(position, half):
    """A corner mask (``corner_mask_<i>``) of +-`half` arcsec around `position`."""
    corners = [
        position.spherical_offsets_by(dx * half * u.arcsec, dy * half * u.arcsec)
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return {
        "ra": [f"{p.ra.deg}deg" for p in corners],
        "dec": [f"{p.dec.deg}deg" for p in corners],
    }


def test_gaia_cutouts_brighter_than_every_stamp_or_masked_are_flagged(
    gaia, tmp_path, caplog
):
    """Star 3, without a stamp, is flagged when brighter than every stamp (a
    star the star fit's peak cut dropped) or masked at its centre (saturated)."""
    table, _ = gaia
    add_star(table, 3, 45.3, 20.6)
    stamps = write_stamps(tmp_path / "s.npz", table)
    psf = joint_psf(tmp_path / "s.npz")
    flag = "Gaia stars [3] have no stamp and are brighter than every stamp"
    build(tmp_path / "a", psf)
    assert not warned(caplog, "have no stamp and are brighter")  # 1e4 DN/int < 6e4
    _, _, extra = load_star_stamps(tmp_path / "s.npz")
    faint = [replace(s, fluence=1e3) for s in stamps]
    save_star_stamps(tmp_path / "f.npz", faint, SUB, **extra)
    build(tmp_path / "b", joint_psf(tmp_path / "f.npz"))
    assert warned(caplog, flag)
    caplog.clear()
    build(tmp_path / "c", psf, corner_mask_1=box_around(catalog(table, 2), 0.03))
    assert warned(caplog, flag)
