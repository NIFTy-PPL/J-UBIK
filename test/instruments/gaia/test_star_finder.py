import sys
import types

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jubik.instruments.gaia.star_finder import load_gaia_stars_in_fov

CORNERS = SkyCoord(
    ra=[10.0, 10.1, 10.1, 10.0], dec=[-5.0, -5.0, -4.9, -4.9], unit="deg"
)


class _FakeJob:
    def __init__(self, table):
        self._table = table

    def get_results(self):
        return self._table


@pytest.fixture
def fake_gaia(monkeypatch):
    """Replace `astroquery.gaia.Gaia` with a stub that records the query."""
    calls = []
    table = Table({"SOURCE_ID": [1, 2], "ra": [10.05, 10.06], "dec": [-4.95, -4.96]})

    class Gaia:
        @staticmethod
        def launch_job_async(query):
            calls.append(query)
            return _FakeJob(table)

    gaia_mod = types.ModuleType("astroquery.gaia")
    gaia_mod.Gaia = Gaia
    pkg = types.ModuleType("astroquery")
    pkg.gaia = gaia_mod
    monkeypatch.setitem(sys.modules, "astroquery", pkg)
    monkeypatch.setitem(sys.modules, "astroquery.gaia", gaia_mod)
    return calls, table


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
