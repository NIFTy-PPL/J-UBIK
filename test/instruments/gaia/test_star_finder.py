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



@pytest.mark.parametrize("ra0", [10.05, 180.0, 0.0])
def test_circle_through_the_corners_when_the_polygon_fails(
    fake_gaia, tmp_path, monkeypatch, ra0
):
    import astroquery.gaia

    calls, table = fake_gaia
    launch = astroquery.gaia.Gaia.launch_job_async

    def polygon_times_out(query):
        if "POLYGON" in query:
            calls.append(query)
            raise RuntimeError("HTTP 500: Job timeout/aborted")
        return launch(query)

    monkeypatch.setattr(astroquery.gaia.Gaia, "launch_job_async", polygon_times_out)
    corners = SkyCoord(
        ra=[ra0 - 0.05, ra0 + 0.05, ra0 + 0.05, ra0 - 0.05],
        dec=[-5.0, -5.0, -4.9, -4.9],
        unit="deg",
    )
    with pytest.warns(UserWarning, match="POLYGON query failed"):
        result = load_gaia_stars_in_fov(corners, str(tmp_path))
    assert len(result) == len(table)
    assert ["POLYGON" in c for c in calls] == [True, False]
    # CIRCLE('ICRS', ra, dec, radius) in degrees.
    ra, dec, radius = map(
        float, calls[1].split("CIRCLE('ICRS',")[1].split(")")[0].split(",")
    )
    center = SkyCoord(ra=ra, dec=dec, unit="deg")
    assert center.separation(SkyCoord(ra=ra0, dec=-4.95, unit="deg")).deg < 1e-3
    assert (center.separation(corners).deg <= radius).all()
    assert radius < 0.08
