# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
import os.path
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import Table


def load_gaia_stars_in_fov(
    fov_corners: SkyCoord,
    library_path: str = "",
    exclude_source_ids: list[int] | None = None,
) -> Table:
    # ----- (A)  filename used for on-disk caching  ---------------------------
    saving_string = "_".join([f"{c.ra.deg:.6f}{c.dec.deg:.6f}" for c in fov_corners])
    if exclude_source_ids:
        saving_string = (
            f"{saving_string}_ex{'_'.join([str(id) for id in exclude_source_ids])}"
        )
    saving_string = os.path.join(library_path, f"{saving_string}.ecsv")

    if library_path != "" and os.path.isfile(saving_string):
        return _with_upper_source_id(Table.read(saving_string))

    # ----- (B)  query the archive  -----------------------------------------
    # Since Gaia archive release 3.15 (2026-10-14) POLYGON queries have run
    # into the server timeout (even the archive's own documentation example)
    # while CIRCLE on the same field answers in seconds.
    try:
        table = query_gaia_polygon(fov_corners, exclude_source_ids)
    except Exception as e:
        warnings.warn(f"Gaia POLYGON query failed ({e!r}), querying a CIRCLE instead")
        table = query_gaia_circle(fov_corners, exclude_source_ids)

    if library_path != "":
        table.write(saving_string, format="ascii.ecsv", overwrite=True)

    return _with_upper_source_id(table)


def _with_upper_source_id(table: Table) -> Table:
    """The archive has returned the id column as `SOURCE_ID` and as `source_id`
    over the years; the star tables read `SOURCE_ID`."""
    if "source_id" in table.colnames:
        table.rename_column("source_id", "SOURCE_ID")
    return table


def query_gaia_polygon(
    fov_corners: SkyCoord, exclude_source_ids: list[int] | None = None
) -> Table:
    """Gaia DR3 sources inside the polygon of the corners, the exact footprint."""
    polygon_str = ", ".join(f"{c.ra.deg}, {c.dec.deg}" for c in fov_corners)
    return _query_gaia(f"POLYGON('ICRS', {polygon_str})", exclude_source_ids)


def query_gaia_circle(
    fov_corners: SkyCoord, exclude_source_ids: list[int] | None = None
) -> Table:
    """Gaia DR3 sources inside the circle through the corners (plus 1%).

    Over-fetches past the corners; the JWST loader drops stars off the
    detector. The center is the mean of the corner unit vectors: an RA mean
    would put a field across RA 0/360 on the far side of the sky.
    """
    center = SkyCoord(fov_corners.cartesian.mean(), frame="icrs")
    radius = center.separation(fov_corners).max().deg * 1.01
    circle_str = f"{center.ra.deg}, {center.dec.deg}, {radius}"
    return _query_gaia(f"CIRCLE('ICRS', {circle_str})", exclude_source_ids)


def _query_gaia(region: str, exclude_source_ids: list[int] | None) -> Table:
    if exclude_source_ids:
        id_list = ", ".join(str(int(sid)) for sid in exclude_source_ids)
        exclusion_clause = f"AND source_id NOT IN ({id_list})"
    else:
        exclusion_clause = ""

    query = f"""
    SELECT
      source_id,
      ra, dec,
      ra_error, dec_error,
      parallax, parallax_error,
      pmra, pmra_error, pmdec, pmdec_error,
      ra_dec_corr, ra_parallax_corr, dec_parallax_corr,
      ra_pmra_corr, ra_pmdec_corr, dec_pmra_corr, dec_pmdec_corr,
      parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
      phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error,
      phot_bp_rp_excess_factor,
      ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
      visibility_periods_used, astrometric_params_solved,
      duplicated_source, phot_variable_flag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {region})
    {exclusion_clause}
    """
    from astroquery.gaia import Gaia

    return Gaia.launch_job_async(query).get_results()
