# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from functools import partial
import os.path

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table, join
from astroquery.gaia import Gaia


def load_gaia_stars_in_fov(fov_corners: SkyCoord, library_path: str = "") -> Table:
    saving_string = "_".join([f"{c.ra.deg:.6f}{c.dec.deg:.6f}" for c in fov_corners])
    saving_string = os.path.join(library_path, f"{saving_string}.ecsv")

    if library_path != "" and os.path.isfile(saving_string):
        return Table.read(saving_string)

    polygon_str = ", ".join([f"{c.ra.deg}, {c.dec.deg}" for c in fov_corners])
    print(polygon_str)
    query = f"""
    SELECT
    source_id, ra, dec
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        POLYGON('ICRS', {polygon_str})
    )
    """
    # AND phot_g_mean_mag < 26
    job = Gaia.launch_job_async(query)
    table: Table = job.get_results()

    if library_path != "":
        table.write(saving_string, format="ascii.ecsv", overwrite=True)

    return table


def join_tables(tables: list[Table]) -> Table:
    out_table = tables[0]
    for table in tables[1:]:
        if len(table) != 0:
            out_table = join(out_table, table, join_type="outer")
    return out_table
