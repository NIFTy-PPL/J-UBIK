import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from jubik.instruments.gaia.star_finder import load_gaia_stars_in_fov

CORNERS = SkyCoord(
    ra=[10.0, 10.1, 10.1, 10.0], dec=[-5.0, -5.0, -4.9, -4.9], unit="deg"
)


def _cache_files(path):
    return sorted(p.name for p in path.iterdir())


@pytest.mark.parametrize("exclude", [None, []])
def test_no_exclusion_default_and_empty(fake_gaia, tmp_path, exclude):
    calls, table = fake_gaia
    result = load_gaia_stars_in_fov(CORNERS, str(tmp_path), exclude_source_ids=exclude)
    assert len(result) == len(table)
    assert len(calls) == 1
    assert "NOT IN" not in calls[0]
    (name,) = _cache_files(tmp_path)
    assert "_ex" not in name
    assert name.endswith(".ecsv")


def test_exclusion_list(fake_gaia, tmp_path):
    calls, _ = fake_gaia
    load_gaia_stars_in_fov(CORNERS, str(tmp_path), exclude_source_ids=[42, 7])
    assert "NOT IN (42, 7)" in calls[0]
    (name,) = _cache_files(tmp_path)
    assert name.endswith("_ex42_7.ecsv")


def test_cache_hit_skips_query(fake_gaia, tmp_path):
    calls, _ = fake_gaia
    first = load_gaia_stars_in_fov(CORNERS, str(tmp_path), exclude_source_ids=[42])
    second = load_gaia_stars_in_fov(CORNERS, str(tmp_path), exclude_source_ids=[42])
    assert len(calls) == 1
    np.testing.assert_array_equal(first["SOURCE_ID"], second["SOURCE_ID"])


def test_missing_library_path_is_created(fake_gaia, tmp_path):
    library = tmp_path / "gaia" / "cache"
    load_gaia_stars_in_fov(CORNERS, str(library))
    (name,) = _cache_files(library)
    assert name.endswith(".ecsv")


def test_no_library_path_does_not_write(fake_gaia, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_gaia_stars_in_fov(CORNERS, "", exclude_source_ids=None)
    assert _cache_files(tmp_path) == []



@pytest.mark.parametrize("ra0", [10.05, 180.0, 0.0])
def test_query_is_a_circle_through_the_corners(fake_gaia, tmp_path, ra0):
    calls, table = fake_gaia
    corners = SkyCoord(
        ra=[ra0 - 0.05, ra0 + 0.05, ra0 + 0.05, ra0 - 0.05],
        dec=[-5.0, -5.0, -4.9, -4.9],
        unit="deg",
    )
    result = load_gaia_stars_in_fov(corners, str(tmp_path))
    assert len(result) == len(table)
    (query,) = calls
    assert "POLYGON" not in query
    # CIRCLE('ICRS', ra, dec, radius) in degrees.
    ra, dec, radius = map(
        float, query.split("CIRCLE('ICRS',")[1].split(")")[0].split(",")
    )
    center = SkyCoord(ra=ra, dec=dec, unit="deg")
    assert center.separation(SkyCoord(ra=ra0, dec=-4.95, unit="deg")).deg < 1e-3
    assert (center.separation(corners).deg <= radius).all()
    assert radius < 0.08


def test_a_release_names_its_table_and_columns(fake_gaia, tmp_path, monkeypatch):
    """A release (as DR4 will be) maps its table and the columns it names
    otherwise: the query selects them under DR3's names, which the star tables
    and the exclusion rules read, and filters by them; its cache is apart."""
    from jubik.instruments.gaia import star_finder

    calls, _ = fake_gaia
    columns = {"ra": "ra_icrs", "ruwe": "ruwe_dr4", "source_id": "id"}
    monkeypatch.setitem(
        star_finder.RELEASES,
        "dr4",
        {"table": "gaiadr4.gaia_source", "columns": columns},
    )
    load_gaia_stars_in_fov(
        CORNERS, str(tmp_path), exclude_source_ids=[7], release="dr4"
    )
    (query,) = calls
    assert "FROM gaiadr4.gaia_source" in query
    assert "ra_icrs AS ra," in query and "ruwe_dr4 AS ruwe," in query
    assert "POINT('ICRS', ra_icrs, dec)" in query and "AND id NOT IN (7)" in query
    (name,) = _cache_files(tmp_path)
    assert name.endswith("_ex7_dr4.ecsv")


def test_dr3_selects_its_columns_by_name_with_the_epoch(fake_gaia, tmp_path):
    from jubik.instruments.gaia.star_finder import COLUMNS

    calls, _ = fake_gaia
    load_gaia_stars_in_fov(CORNERS, str(tmp_path))
    (query,) = calls
    assert "FROM gaiadr3.gaia_source" in query and " AS " not in query
    select = query.split("SELECT")[1].split("FROM")[0]
    assert [c.strip() for c in select.split(",")] == list(COLUMNS)
    assert "ref_epoch" in COLUMNS


def test_an_unknown_release_raises(fake_gaia, tmp_path):
    with pytest.raises(ValueError, match="unknown Gaia release 'dr9'"):
        load_gaia_stars_in_fov(CORNERS, str(tmp_path), release="dr9")
