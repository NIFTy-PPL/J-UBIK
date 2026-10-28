from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.table import MaskedColumn, Table
from astropy.time import Time

from jubik.instruments.jwst.alignment.star_alignment import StarTables
from jubik.instruments.jwst.data.preloader.preloader import _exposure_start_time
from test.conftest import GAIA_CORRELATIONS

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
            "ra_error": np.ones(n),
            "dec_error": np.ones(n),
            "pmra_error": np.ones(n),
            "pmdec_error": np.ones(n),
            **{name: np.zeros(n) for name in GAIA_CORRELATIONS},
        },
        units={
            "ra": "deg",
            "dec": "deg",
            "pmra": "mas / yr",
            "pmdec": "mas / yr",
            "parallax": "mas",
            "ra_error": "mas",
            "dec_error": "mas",
            "pmra_error": "mas / yr",
            "pmdec_error": "mas / yr",
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


def test_position_covariance_propagates_gaia_covariance_to_the_epoch():
    table = _table([1])
    sra, sdec, spmra, spmdec = 0.3, 0.1, 0.4, 0.2
    rho = dict(
        ra_dec=0.2,
        ra_pmra=0.3,
        ra_pmdec=-0.1,
        dec_pmra=0.15,
        dec_pmdec=-0.25,
        pmra_pmdec=0.4,
    )
    table["ra_error"][0], table["pmra_error"][0] = sra, spmra
    table["dec_error"][0], table["pmdec_error"][0] = sdec, spmdec
    for pair, value in rho.items():
        table[f"{pair}_corr"][0] = value
    dt = 5
    (star,) = StarTables([table], [Time("J2016.0") + dt * u.yr]).get_stars(0)
    # (RA cos(dec), Dec) at ra + dt * pmra, dec + dt * pmdec, in mas^2
    ra = sra**2 + 2 * dt * rho["ra_pmra"] * sra * spmra + dt**2 * spmra**2
    dec = sdec**2 + 2 * dt * rho["dec_pmdec"] * sdec * spmdec + dt**2 * spmdec**2
    ra_dec = (
        rho["ra_dec"] * sra * sdec
        + dt * rho["ra_pmdec"] * sra * spmdec
        + dt * rho["dec_pmra"] * sdec * spmra
        + dt**2 * rho["pmra_pmdec"] * spmra * spmdec
    )
    np.testing.assert_allclose(
        star.position_covariance * 1e6, [[ra, ra_dec], [ra_dec, dec]], rtol=1e-6
    )


def test_table_without_error_columns_gives_no_covariance():
    table = _table([1])
    table.remove_columns([c for c in table.colnames if c.endswith(("_error", "_corr"))])
    (star,) = StarTables([table], [T_OBS]).get_stars(0)
    assert star.position_covariance is None


def test_masked_gaia_error_makes_the_covariance_nan_and_keeps_the_star():
    table = _table([1, 2])
    table["pmra_error"] = MaskedColumn(table["pmra_error"], mask=[False, True])
    stars = StarTables([table], [T_OBS]).get_stars(0)
    assert [int(s.id) for s in stars] == [1, 2]
    assert np.isfinite(stars[0].position_covariance).all()
    assert np.isnan(stars[1].position_covariance).any()


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
