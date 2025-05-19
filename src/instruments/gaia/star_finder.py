# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from functools import partial
import os.path

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia


def load_gaia_stars_in_fov(
    fov_corners: SkyCoord,
    library_path: str = "",
    exclude_source_ids: list[int] | None = None,
) -> Table:
    # ----- (A)  filename used for on-disk caching  ---------------------------
    saving_string = "_".join([f"{c.ra.deg:.6f}{c.dec.deg:.6f}" for c in fov_corners])
    if exclude_source_ids is not None or exclude_source_ids != []:
        saving_string = (
            f"{saving_string}_ex{'_'.join([str(id) for id in exclude_source_ids])}"
        )
    saving_string = os.path.join(library_path, f"{saving_string}.ecsv")

    if library_path != "" and os.path.isfile(saving_string):
        return Table.read(saving_string)

    # ----- (B)  build the ADQL query  ---------------------------------------
    polygon_str = ", ".join([f"{c.ra.deg}, {c.dec.deg}" for c in fov_corners])

    # build the optional exclusion clause
    if exclude_source_ids:
        # make a comma-separated string of literal integers
        id_list = ", ".join(str(int(sid)) for sid in exclude_source_ids)
        exclusion_clause = f"AND source_id NOT IN ({id_list})"
    else:
        exclusion_clause = ""  # nothing to add

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
    WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            POLYGON('ICRS', {polygon_str})
          )
    {exclusion_clause}
    """

    # query = f"""
    # SELECT
    # source_id, ra, dec
    # FROM gaiadr3.gaia_source
    # WHERE 1 = CONTAINS(
    #     POINT('ICRS', ra, dec),
    #     POLYGON('ICRS', {polygon_str})
    # )
    # """

    job = Gaia.launch_job_async(query)
    table: Table = job.get_results()

    if library_path != "":
        table.write(saving_string, format="ascii.ecsv", overwrite=True)

    return table
