import numpy as np
from astropy.table import Table

from jubik.instruments.jwst.alignment.star_alignment import StarTables

T_OBS = "2023-01-01T00:00:00"


def _table(ids, parallax=None):
    n = len(ids)
    return Table(
        {
            "SOURCE_ID": np.asarray(ids, dtype=np.int64),
            "ra": np.full(n, 10.0) + 1e-3 * np.arange(n),
            "dec": np.full(n, -5.0),
            "pmra": np.zeros(n),
            "pmdec": np.zeros(n),
            "parallax": np.ones(n) if parallax is None else np.asarray(parallax),
        },
        units={
            "ra": "deg",
            "dec": "deg",
            "pmra": "mas / yr",
            "pmdec": "mas / yr",
            "parallax": "mas",
        },
    )


def test_empty_table_gives_no_stars():
    tables = StarTables([_table([])], [T_OBS])
    assert tables.get_stars(0) == []
    assert tables.get_stars() == []


def test_all_empty_tables_joined():
    tables = StarTables([_table([]), _table([])], [T_OBS, T_OBS])
    assert tables.get_stars() == []


def test_one_star():
    tables = StarTables([_table([11])], [T_OBS])
    stars = tables.get_stars(0)
    assert [s.id for s in stars] == [11]


def test_duplicate_ids_across_observations_joined_once():
    tables = StarTables([_table([1, 2]), _table([2, 3])], [T_OBS, T_OBS])
    assert sorted(s.id for s in tables.get_stars()) == [1, 2, 3]
    assert [s.id for s in tables.get_stars(1)] == [2, 3]


def test_negative_and_missing_parallax_keep_the_star():
    table = _table([1, 2, 3], parallax=[1.0, -0.3, np.nan])
    tables = StarTables([table], [T_OBS])
    assert sorted(int(s.id) for s in tables.get_stars(0)) == [1, 2, 3]


def test_missing_proper_motion_drops_the_star():
    table = _table([1, 2])
    table["pmra"][1] = np.nan
    tables = StarTables([table], [T_OBS])
    assert [int(s.id) for s in tables.get_stars(0)] == [1]
