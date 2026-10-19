from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from jubik.instruments.jwst.alignment.star_alignment import StarTables
from jubik.instruments.jwst.data.preloader.preloader import _exposure_start_time

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


def _fake_dm(created, date_beg=None, expstart_mjd=None):
    # Mimics stdatamodels: unset metadata reads as None.
    return SimpleNamespace(
        meta=SimpleNamespace(
            date=created,
            observation=SimpleNamespace(date_beg=date_beg),
            exposure=SimpleNamespace(start_time=expstart_mjd),
        )
    )


def _moving_star_position(epoch):
    table = _table([1])
    table["dec"][0] = 0.0
    table["pmdec"][0] = 100.0  # mas / yr
    return StarTables([table], [epoch]).get_stars(0)[0].position


def test_epoch_ignores_file_creation_date():
    a = _exposure_start_time(_fake_dm("2020-01-01T00:00:00", "2022-07-01T12:00:00"))
    b = _exposure_start_time(_fake_dm("2026-01-01T00:00:00", "2022-07-01T12:00:00"))
    assert a.scale == "utc"
    assert _moving_star_position(a).separation(_moving_star_position(b)) == 0


def test_epoch_falls_back_to_expstart_mjd():
    t = _exposure_start_time(_fake_dm("2026-01-01T00:00:00", expstart_mjd=59761.5))
    assert t.scale == "utc"
    assert t.isot == "2022-07-01T12:00:00.000"


def test_two_exposure_epochs_differ_by_pm_times_dt():
    t1 = _exposure_start_time(_fake_dm("2026-01-01", "2022-01-01T00:00:00"))
    t2 = _exposure_start_time(_fake_dm("2026-01-01", "2024-01-01T00:00:00"))
    dt_yr = (t2 - t1).to_value("day") / 365.25  # 730 days
    sep = _moving_star_position(t1).separation(_moving_star_position(t2))
    np.testing.assert_allclose(sep.to_value("mas"), 100.0 * dt_yr, rtol=1e-6)


def test_missing_exposure_time_raises():
    with pytest.raises(ValueError, match="No exposure start time"):
        _exposure_start_time(_fake_dm("2026-01-01T00:00:00"))
