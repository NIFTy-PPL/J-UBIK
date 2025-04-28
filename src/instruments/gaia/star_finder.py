# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from functools import partial
from os.path import join, isfile

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia


def load_gaia_stars_in_fov(fov_corners: SkyCoord, library_path: str = "") -> Table:
    saving_string = "_".join([f"{c.ra.deg:.6f}{c.dec.deg:.6f}" for c in fov_corners])
    saving_string = join(library_path, f"{saving_string}.ecsv")

    if library_path != "" and isfile(saving_string):
        return Table.read(saving_string)

    polygon_str = ", ".join([f"{c.ra.deg}, {c.dec.deg}" for c in fov_corners])
    query = f"""
    SELECT
    source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        POLYGON('ICRS', {polygon_str})
    )
    AND phot_g_mean_mag < 26
    """
    job = Gaia.launch_job_async(query)
    table: Table = job.get_results()

    if library_path != "":
        table.write(saving_string, format="ascii.ecsv", overwrite=True)

    return table
