"""``psf.joint_stars``' ``tied_position_std``: the stamps of one star share its
sky position and sit there moved by their exposure's shift, the lens's."""

from dataclasses import replace

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.psf_correction import StarStamp, save_star_stamps

from test_brighter_fatter import BAND, DN_PER_UNIT, latents
from test_joint_gaia_stars import (
    FILES,
    GRID_PIXEL,
    H,
    SHIFT_STD,
    EpochJwstData,
    add_star,
    exposure_wcs,
    gaia,  # noqa: F401, the fixture
    gaia_columns,
    stamp_sites,
    standalone,
    true_position,
    warned,
)
from test_joint_gaia_stars import build as build_gaia
from test_joint_gaia_stars import write_stamps as write_gaia_stamps
from test_joint_psf_stars import SUB, base_kernel, energy, fit, joint_psf, tree
from test_jwst_likelihoods import JWST_DATA_SITES, SKY_DOMAIN, make_config, make_grid
from test_jwst_psf_broadening import wide_build_webb_psf
from test_psf_correction import SIZE

STD = 0.05  # arcsec, the tie's position prior
PAD = SUB * 12  # the star model's canvas: only the positions' linearization differs
CANVAS = (SUB * SIZE + 2 * PAD, PAD)

# stamp i: (star, file, the star's detector (x, y) there); the centre pixel is
# its rounding. Stars 0 and 1 in both files, star 2 in fake_1 alone, star 0
# once more in a file outside the fit.
SITES = [
    (0, "fake_0.fits", 20.3, 24.6),
    (1, "fake_0.fits", 40.4, 15.2),
    (0, "fake_1.fits", 20.6, 24.2),
    (1, "fake_1.fits", 39.7, 15.6),
    (2, "fake_1.fits", 30.2, 40.4),
    (0, "other.fits", 20.3, 24.6),
]


@pytest.fixture
def field(monkeypatch):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, EpochJwstData)
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", wide_build_webb_psf)


def write(path, sites=SITES, data=None, seed=0):
    """The stamps at `sites` with `data` (default: noise around a bump), each
    with its star, file, centre pixel and that pixel's RA/Dec."""
    rng = np.random.default_rng(seed)
    yx = np.mgrid[:SIZE, :SIZE] - H
    if data is None:
        bump = 50 * np.exp(-0.5 * (yx**2).sum(0) / 2.0)
        data = [bump + 5 + rng.normal(size=(SIZE, SIZE)) for _ in sites]
    mask = np.ones((SIZE, SIZE), bool)
    mask[0, :3] = False
    dn = [dict(zip(FILES, DN_PER_UNIT)).get(site[1], 250.0) for site in sites]
    stamps = [
        StarStamp(d, np.full((SIZE, SIZE), 0.5 + 0.1 * i), mask, base_kernel(), f, u_)
        for i, (d, u_, f) in enumerate(
            zip(data, dn, [1e4 * (i + 1) for i in range(len(sites))])
        )
    ]
    star, file, x, y = (np.array(c) for c in zip(*sites))
    x, y = np.round(x.astype(float)), np.round(y.astype(float))
    sky = exposure_wcs().pixel_to_world(x, y)
    save_star_stamps(
        path, stamps, SUB, star=star.astype(int), exposure=file, x=x, y=y,
        ra=sky.ra.deg, dec=sky.dec.deg,
    )  # fmt: skip
    return stamps


def tied(path, std=STD, **options):
    entry = joint_psf(path, **options)["joint_stars"][BAND] | {"tied_position_std": std}
    return {"pixel_box": True, "joint_stars": {BAND: entry}}


def build(tmp_path, psf):
    tmp_path.mkdir(exist_ok=True)
    cfg = make_config(tmp_path, n_files=2)
    cfg["telescope"]["psf"] |= dict(psf)
    return build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def tied_position(x, sites, j, k):
    """Stamp `j`'s star position, subsampled (x, y) of the stamp, computed
    apart from the code: the mean of its star's centres in the fit's files
    (RA and Dec averaged in degrees), moved on the sky by ``STD`` times its
    latent (row `k` of the stars), then on the sky grid against its exposure's
    shift, into the exposure's pixels."""
    star, file = sites[j][:2]
    centres = [
        exposure_wcs().pixel_to_world(*np.round(np.array(s[2:], float)))
        for s in sites
        if s[0] == star and s[1] in FILES
    ]
    mean = SkyCoord(
        np.mean([c.ra.deg for c in centres]),
        np.mean([c.dec.deg for c in centres]),
        unit="deg",
    )
    offset = STD * np.asarray(x[f"{BAND}_tied_star_position"])[k]
    position = mean.spherical_offsets_by(offset[0] * u.arcsec, offset[1] * u.arcsec)
    grid = make_grid().spatial
    row, column = grid.world_to_indices_yx(position)
    shift = SHIFT_STD * np.asarray(x[f"{BAND}_shift"])[FILES.index(file)] / GRID_PIXEL
    moved = grid.indices_yx_to_world(row - shift[0], column - shift[1])
    corner = np.round(np.array(sites[j][2:], float)) - H
    xy = np.array(exposure_wcs().world_to_pixel(moved), float) - corner
    return (xy + 0.5) * SUB - 0.5


def test_a_stars_stamps_share_its_position_moved_by_the_shift(field, tmp_path, caplog):
    """Each stamp in the fit's files sits at its star's mean centre moved by
    the star's latent and its exposure's shift; the stamp outside them keeps a
    free position: the energy is an independent model's."""
    stamps = write(tmp_path / "s.npz")
    products = build(tmp_path / "b", tied(tmp_path / "s.npz"))
    (stars,) = products.target.joint_stars
    domain = tree(stars.domain)
    assert domain[f"{BAND}_tied_star_position"].shape == (3, 2)
    assert domain[f"{BAND}_star_position"].shape == (1, 2)  # other.fits
    assert f"{BAND}_shift" in domain
    assert warned(caplog, "stamps of ['other.fits'], no file of the fit, keep free")
    assert "5 star stamps of 3 stars tied to the exposure shifts" in caplog.text

    x = latents(products)
    positions = [tied_position(x, SITES, j, SITES[j][0]) for j in range(5)]
    want = standalone(stamps, x, f"{BAND}_", positions + [None], canvas=CANVAS)
    assert np.isclose(energy(stars, x), want, rtol=2e-8)


def test_tied_stamps_have_no_position_of_their_own(field, tmp_path):
    """Tied, the per-stamp position latents are gone; untied, every stamp has one
    and the tie's latent is absent."""
    write(tmp_path / "s.npz", sites=SITES[:5])
    (on,) = build(tmp_path / "a", tied(tmp_path / "s.npz")).target.joint_stars
    (off,) = build(tmp_path / "b", joint_psf(tmp_path / "s.npz")).target.joint_stars
    assert f"{BAND}_star_position" not in tree(on.domain)
    assert tree(off.domain)[f"{BAND}_star_position"].shape == (5, 2)
    assert f"{BAND}_tied_star_position" not in tree(off.domain)
    assert f"{BAND}_shift" not in tree(off.domain)


def test_without_the_tie_nothing_changes(field, tmp_path):
    """``tied_position_std: null`` is the entry without it: the same energy and
    gradient, bit for bit (today's free stamps, pinned in test_joint_*)."""
    write(tmp_path / "s.npz")
    a = build(tmp_path / "a", joint_psf(tmp_path / "s.npz"))
    b = build(tmp_path / "b", tied(tmp_path / "s.npz", std=None))
    x = latents(a)
    la, lb = a.target.likelihood, b.target.likelihood
    assert float(la(jft.Vector(x))) == float(lb(jft.Vector(x)))
    ga = jax.grad(lambda y: la(jft.Vector(y)))(x)
    gb = jax.grad(lambda y: lb(jft.Vector(y)))(x)
    assert ga.keys() == gb.keys()
    for key in ga:
        np.testing.assert_array_equal(ga[key], gb[key], err_msg=key)


def test_the_lens_and_the_stars_share_the_shift(field, tmp_path):
    """The shift's gradient of the whole likelihood is the lens's plus the tied
    stars', and both are nonzero: one latent moves both."""
    write(tmp_path / "s.npz")
    products = build(tmp_path / "b", tied(tmp_path / "s.npz"))
    x = latents(products)
    (stars,) = products.target.joint_stars
    lens = replace(products.target, joint_stars=[]).likelihood

    def grad(likelihood):
        return np.asarray(
            jax.grad(lambda y: likelihood(jft.Vector(y)))(x)[f"{BAND}_shift"]
        )

    total, parts = grad(products.target.likelihood), [grad(lens), grad(stars)]
    assert all(np.any(p != 0) for p in parts)
    np.testing.assert_allclose(total, sum(parts), rtol=1e-10)


def test_the_stars_alone_recover_injected_shifts(field, tmp_path):
    """Stamps made by the model at known shifts (40, -30 and -20, 50 mas) with
    noise: the tied stars alone recover the shift between the exposures. Their
    common shift is the gauge of the stars' positions, so only the difference is
    the stars'. The data come from the model under test: this checks the fit,
    `test_a_stars_stamps_share_its_position_moved_by_the_shift` the placement."""
    sites = SITES[:5]
    stamps = write(tmp_path / "s.npz", sites=sites)
    psf = tied(tmp_path / "s.npz", kappa=None)
    (stars,) = build(tmp_path / "a", psf).target.joint_stars
    truth = dict(tree(jft.random_like(jax.random.PRNGKey(11), stars.domain)))
    shift = np.array([[0.4, -0.3], [-0.2, 0.5]])
    truth[f"{BAND}_shift"] = shift
    truth[f"{BAND}_star_flux"] = np.full(len(sites), 3.0)  # bright: sub-mas positions
    data, std = np.array([s.data for s in stamps]), np.array([s.std for s in stamps])
    mask = np.array([s.mask for s in stamps])
    residual = np.asarray(stars.normalized_residual(jft.Vector(truth)))
    noise = np.random.default_rng(3).normal(size=residual.shape)
    data[mask] = data[mask] + std[mask] * (noise - residual)  # model + noise
    write(tmp_path / "s.npz", sites=sites, data=list(data))
    (stars,) = build(tmp_path / "b", psf).target.joint_stars

    learned = fit(stars)
    difference = np.diff(np.asarray(learned[f"{BAND}_shift"]), axis=0)
    error = 1e3 * SHIFT_STD * np.abs(difference - np.diff(shift, axis=0)).max()
    assert error < 0.2, error  # mas, 0.02 here; the difference is 60 and 80 mas


def test_a_gaia_star_keeps_gaias_prior_and_the_others_are_tied(gaia, tmp_path, caplog):
    """With gaia_alignment, the stamps at Gaia stars 1 and 2 sit at Gaia's
    positions (their astrometry latents), the stamp at no Gaia star is tied, the
    one outside the fit's files stays free: an independent model's energy."""
    table, _ = gaia
    stamps = write_gaia_stamps(tmp_path / "g.npz", table)
    sites = stamp_sites(table)
    stars = [0, 1, 1, 0, 2, 0]  # star 2: the stamp at no Gaia star
    save_star_stamps(
        tmp_path / "s.npz", stamps, SUB, star=np.array(stars),
        exposure=np.array([s[0] for s in sites]),
        x=np.array([s[1] for s in sites]), y=np.array([s[2] for s in sites]),
        ra=exposure_wcs().pixel_to_world([s[1] for s in sites], [s[2] for s in sites]).ra.deg,
        dec=exposure_wcs().pixel_to_world([s[1] for s in sites], [s[2] for s in sites]).dec.deg,
    )  # fmt: skip
    products = build_gaia(tmp_path / "b", tied(tmp_path / "s.npz"))
    (term,) = products.target.joint_stars
    domain = tree(term.domain)
    assert {"star_1_astrometry", "star_2_astrometry"} <= set(domain)
    assert domain[f"{BAND}_tied_star_position"].shape == (1, 2)
    assert domain[f"{BAND}_star_position"].shape == (1, 2)
    outside = [r for r in caplog.records if "no file of the fit" in r.getMessage()]
    assert len(outside) == 1  # the Gaia tie's warning, not the shifts' tie's too

    x = latents(products)
    wcs = exposure_wcs()
    corner = [np.array(s[1:]) - H for s in sites]
    positions = [
        true_position(table, i, x, e, wcs, corner[j])
        for j, (i, e) in enumerate([(0, 1), (1, 0), (1, 1), (0, 0)])
    ]
    lone = [(2, "fake_0.fits", *sites[4][1:])]
    positions.append(tied_position(x, lone, 0, 0))
    want = standalone(stamps, x, f"{BAND}_", positions + [None], canvas=CANVAS)
    assert np.isclose(energy(term, x), want, rtol=2e-8)


def test_a_gaia_star_only_the_astrometry_rules_drop_is_tied(gaia, tmp_path, caplog):
    """Tied, Gaia star 2, whose RUWE the rules reject, keeps its stamps, tied to
    the shifts without Gaia's prior; star 1, a galaxy by the rules, goes with its
    stamps as without the tie; star 3, RUWE too but without a stamp, goes with
    its cutouts and is logged."""
    table, _ = gaia
    add_star(table, 3, 45.3, 20.6)
    gaia_columns(
        table, ruwe=[1.0, 2.0, 2.0], classprob_dsc_combmod_galaxy=[0.9, 0.0, 0.0]
    )
    stamps = write_gaia_stamps(tmp_path / "g.npz", table, fluence=[1e3] * 6)
    sites = stamp_sites(table)
    x_, y_ = [s[1] for s in sites], [s[2] for s in sites]
    sky = exposure_wcs().pixel_to_world(x_, y_)
    save_star_stamps(
        tmp_path / "s.npz", stamps, SUB, star=np.array([0, 1, 1, 0, 2, 0]),
        exposure=np.array([s[0] for s in sites]), x=np.array(x_), y=np.array(y_),
        ra=sky.ra.deg, dec=sky.dec.deg,
    )  # fmt: skip
    psf = tied(tmp_path / "s.npz")
    psf["joint_stars"][BAND]["exclude"] = True
    (term,) = build_gaia(tmp_path / "b", psf).target.joint_stars
    domain = tree(term.domain)
    assert "star_2_astrometry" not in domain and "star_1_astrometry" not in domain
    # star 2's two stamps and the lone stamp: two tied stars; star 1's two dropped
    assert domain[f"{BAND}_tied_star_position"].shape == (2, 2)
    assert domain[f"{BAND}_star_flux"].shape == (4,)
    assert (
        "the stamps of Gaia stars [2], which only the astrometry rules" in caplog.text
    )
    assert warned(caplog, "Gaia stars excluded: 1: galaxy probability 0.9")
    assert not warned(caplog, "2: RUWE")  # its stamps stay
    assert warned(caplog, "3: RUWE 2")  # its cutouts go
    assert not warned(caplog, "only the lens pins")  # the tied stamps pin them too


def test_the_tie_needs_each_stamps_star_and_a_positive_std(field, tmp_path):
    stamps = write(tmp_path / "s.npz", sites=SITES[:5])
    file, x, y = (np.array(c) for c in zip(*[s[1:] for s in SITES[:5]]))
    save_star_stamps(
        tmp_path / "bare.npz", stamps, SUB, exposure=file, x=x, y=y, ra=x, dec=y
    )
    with pytest.raises(
        ValueError, match=r"tied_position_std needs the stamps' \['star'\]"
    ):
        build(tmp_path / "a", tied(tmp_path / "bare.npz"))
    with pytest.raises(ValueError, match="tied_position_std must be > 0"):
        build(tmp_path / "b", tied(tmp_path / "s.npz", std=0.0))
