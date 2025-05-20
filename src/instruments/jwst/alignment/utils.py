import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


# ------------------------------------------------------------------
#  Small helpers
# ------------------------------------------------------------------
def _flat_sky_sep(p1: SkyCoord, p2: SkyCoord) -> u.Quantity:
    """
    “flat” separation: treat the sky as a flat plane that is tangent
    at p1.  Good only for small angles.

    Returned in arc-sec.
    """
    p1, p2 = p1.icrs, p2.icrs
    dra = (p2.ra - p1.ra).wrap_at(180 * u.deg)  #  –180 … +180
    ddec = p2.dec - p1.dec

    dx = (dra.to(u.rad) * np.cos(p1.dec)).to(u.rad)
    dy = ddec.to(u.rad)

    return np.hypot(dx, dy).to(u.arcsec)  # √(dx²+dy²)


def _true_sep(p1: SkyCoord, p2: SkyCoord) -> u.Quantity:
    """Exact great-circle separation (arc-sec)."""
    return p1.icrs.separation(p2.icrs).to(u.arcsec)


def _true_sep2(p1: SkyCoord, p2: SkyCoord) -> u.Quantity:
    """
    Exact great-circle separation between two sky positions
    (returned as an astropy Quantity in arc-seconds),
    but without using SkyCoord.separation() internally.

    Uses the numerically stable haversine formula:

        Δσ = 2 · arcsin( √( hav(Δδ) + cos δ1 · cos δ2 · hav(Δα) ) )

    where hav(x) = sin²(x/2).
    """

    # 1. extract long/lat in radians
    ra1 = p1.icrs.ra.to_value(u.rad)
    dec1 = p1.icrs.dec.to_value(u.rad)
    ra2 = p2.icrs.ra.to_value(u.rad)
    dec2 = p2.icrs.dec.to_value(u.rad)

    # 2. haversine of longitude/latitude differences
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1

    sin2_ddec = np.sin(d_dec / 2.0) ** 2
    sin2_dra = np.sin(d_ra / 2.0) ** 2

    # 3. haversine formula
    haversine = sin2_ddec + np.cos(dec1) * np.cos(dec2) * sin2_dra
    ang_rad = 2.0 * np.arcsin(np.sqrt(haversine))

    # 4. return as arc-seconds
    return (ang_rad * u.rad).to(u.arcsec)


# ------------------------------------------------------------------
#  PUBLIC ROUTINE
# ------------------------------------------------------------------
def sky_offset_diagnostics(
    pos1: SkyCoord,
    pos2: SkyCoord,
    dra_shift: u.Quantity = 0 * u.mas,
    ddec_shift: u.Quantity = 0 * u.mas,
):
    """
    1)  Compute the great-circle (“true”) separation σ₀ between pos1 ↔ pos2.
    2)  Compute the flat-sky (“flat”) separation σ₀^flat .
        ε₀  = |σ₀ − σ₀^flat|   (small-angle error)
    3)  Optionally shift pos1 by (dra_shift, ddec_shift) and repeat.
        Returns σ₁, σ₁^flat, ε₁ and the deltas with respect to the baseline.

    All angles are `astropy.units.Quantity`; defaults mean “no shift”.

    Returns
    -------
    dict with (always)
        'sigma0'     : true separation        (arc-sec)
        'flat0'      : flat separation       (arc-sec)
        'eps0'       : |sigma0 - flat0|       (mas)

      and, if a non-zero shift was supplied,
        'sigma1', 'flat1', 'eps1'             : as above, after the shift
        'delta_sigma'                         : sigma1 − sigma0 (mas)
        'delta_eps'                           : eps1  − eps0  (mas)
    """

    # ---------------- baseline ------------------------------------
    sigma0 = _true_sep(pos1, pos2)
    assert np.isclose(sigma0.to(u.rad).value, _true_sep2(pos1, pos2).to(u.rad).value)
    flat0 = _flat_sky_sep(pos1, pos2)
    eps0 = (sigma0 - flat0).to(u.mas)

    out = dict(sigma0=sigma0, flat0=flat0, eps0=eps0)

    # --------------- optional: shift pos1 -------------------------
    if dra_shift != 0 * u.mas or ddec_shift != 0 * u.mas:
        pos1_shift = pos1.spherical_offsets_by(
            dra_shift.to(u.rad), ddec_shift.to(u.rad)
        )

        sigma1 = _true_sep(pos1_shift, pos2)
        flat1 = _flat_sky_sep(pos1_shift, pos2)
        eps1 = (sigma1 - flat1).to(u.mas)

        out.update(
            sigma1=sigma1,
            flat1=flat1,
            eps1=eps1,
            delta_sigma=(sigma1 - sigma0).to(u.mas),
            delta_eps=(eps1 - eps0).to(u.mas),
        )

    return out


def some_evaluation(index, jwst_data, star_tables):
    import matplotlib.pyplot as plt
    from functools import partial
    from ..plotting.plotting_sky import plot_jwst_panels, plot_sky_coords
    from .star_alignment import Star

    from astropy.time import Time

    g2016 = Time("J2016.0")

    newstars = star_tables.get_stars(index) if star_tables else None

    pos = newstars[0].position
    pos = SkyCoord(ra=pos.ra, dec=pos.dec)
    bor = jwst_data.get_boresight_world_coords()
    results_bor = sky_offset_diagnostics(
        bor, pos, dra_shift=0.3 * u.arcsec, ddec_shift=0.3 * u.arcsec
    )

    pixdist = jwst_data.meta.pixel_distance.to(u.mas)
    for star in newstars:
        pos = SkyCoord(ra=star.position.ra, dec=star.position.dec)
        bor = jwst_data.get_boresight_world_coords()
        crval = jwst_data.get_reference_pixel_world_coords()
        results_bor = sky_offset_diagnostics(
            bor, pos, dra_shift=0.3 * u.arcsec, ddec_shift=0.3 * u.arcsec
        )
        results_crval = sky_offset_diagnostics(
            crval, pos, dra_shift=0.3 * u.arcsec, ddec_shift=0.3 * u.arcsec
        )

        print(
            pixdist,
            f"|true-flat|: {results_bor['eps0']:.3f}",
            f"|true-flat|-shifted: {results_bor['eps1']:.3f}",
            f"delta |true-flat| - |true-flat|-shifted : {results_bor['delta_eps']:.3f}",
        )

    print()

    pixdist = jwst_data.meta.pixel_distance.to(u.mas)
    for star in newstars:
        pos = SkyCoord(ra=star.position.ra, dec=star.position.dec)
        bor = jwst_data.get_boresight_world_coords()
        crval = jwst_data.get_reference_pixel_world_coords()
        results_bor = sky_offset_diagnostics(
            bor, pos, dra_shift=0.3 * u.arcsec, ddec_shift=0.3 * u.arcsec
        )
        results_crval = sky_offset_diagnostics(
            crval, pos, dra_shift=0.3 * u.arcsec, ddec_shift=0.3 * u.arcsec
        )

        print(
            pixdist,
            f"|true-flat|: {results_crval['eps0']:.3f}",
            f"|true-flat|-shifted: {results_crval['eps1']:.3f}",
            f"delta |true-flat| - |true-flat|-shifted : {results_crval['delta_eps']:.3f}",
        )

    oldstars = [
        Star(id=star.id, position=star.position.apply_space_motion(g2016))
        for star in newstars
    ]

    def plot_multiple(ii, ax):
        applies = [
            partial(
                plot_sky_coords,
                sky_coords=[s.position for s in oldstars],
                marker_color="red",
                marker="x",
            ),
            partial(
                plot_sky_coords,
                sky_coords=[s.position for s in newstars],
                marker_color="orange",
                marker="x",
            ),
        ]
        for apply in applies:
            apply(ii, ax)

    fig, axes = plot_jwst_panels(
        [jwst_data.dm.data],
        [jwst_data.wcs],
        nrows=1,
        ncols=1,
        vmin=0.05,
        vmax=0.5,
        # vmin=220.05,
        # vmax=230.5,
        coords_plotter=plot_multiple,
    )
    plt.show()
