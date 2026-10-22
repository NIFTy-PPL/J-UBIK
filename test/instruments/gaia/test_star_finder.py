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


def test_no_library_path_does_not_write(fake_gaia, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_gaia_stars_in_fov(CORNERS, "", exclude_source_ids=None)
    assert _cache_files(tmp_path) == []


def _circle(query):
    """(ra, dec, radius) in degrees of the query's CIRCLE('ICRS', ...)."""
    return map(float, query.split("CIRCLE('ICRS',")[1].split(")")[0].split(","))


def test_polygon_first(fake_gaia, tmp_path):
    calls, _ = fake_gaia
    load_gaia_stars_in_fov(CORNERS, str(tmp_path))
    assert len(calls) == 1 and "POLYGON('ICRS', 10.0, -5.0, 10.1, -5.0" in calls[0]
    assert "CIRCLE" not in calls[0]


def test_circle_through_the_corners_when_the_polygon_fails(fake_polygon_timeout, tmp_path):
    calls, table = fake_polygon_timeout
    result = load_gaia_stars_in_fov(CORNERS, str(tmp_path))
    assert len(result) == len(table)
    assert ["POLYGON" in c for c in calls] == [True, False]
    # CIRCLE('ICRS', ra, dec, radius) in degrees, centered on the corners and reaching them all.
    ra, dec, radius = _circle(calls[1])
    center = SkyCoord(ra=ra, dec=dec, unit="deg")
    assert np.allclose([ra, dec], [10.05, -4.95], atol=1e-3)
    assert (center.separation(CORNERS).deg <= radius).all()
    assert radius < 0.08


def test_lowercase_source_id_is_renamed(fake_gaia, tmp_path):
    _, table = fake_gaia
    table.rename_column("SOURCE_ID", "source_id")
    fresh = load_gaia_stars_in_fov(CORNERS, str(tmp_path))
    cached = load_gaia_stars_in_fov(CORNERS, str(tmp_path))
    assert "SOURCE_ID" in fresh.colnames and "SOURCE_ID" in cached.colnames


@pytest.mark.parametrize("ra0", [180.0, 0.0])
def test_circle_center_across_ra_wrap(fake_polygon_timeout, tmp_path, ra0):
    calls, _ = fake_polygon_timeout
    corners = SkyCoord(
        ra=[ra0 - 0.05, ra0 + 0.05, ra0 + 0.05, ra0 - 0.05],
        dec=[-5.0, -5.0, -4.9, -4.9],
        unit="deg",
    )
    load_gaia_stars_in_fov(corners, str(tmp_path))
    ra, dec, radius = _circle(calls[1])
    center = SkyCoord(ra=ra, dec=dec, unit="deg")
    assert center.separation(SkyCoord(ra=ra0, dec=-4.95, unit="deg")).deg < 1e-3
    assert (center.separation(corners).deg <= radius).all()
    assert radius < 0.08


def test_both_regions_failing_raises(fake_polygon_timeout, tmp_path, monkeypatch):
    import astroquery.gaia

    def fail(query):
        raise RuntimeError("408")

    monkeypatch.setattr(astroquery.gaia.Gaia, "launch_job_async", fail)
    with pytest.raises(RuntimeError, match="408"):
        load_gaia_stars_in_fov(CORNERS, str(tmp_path))
    assert _cache_files(tmp_path) == []
