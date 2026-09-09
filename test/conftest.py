import sys
import types

import pytest
from astropy import units as u
from astropy.table import Table


class _FakeJob:
    def __init__(self, table):
        self._table = table

    def get_results(self):
        return self._table


@pytest.fixture
def fake_gaia(monkeypatch):
    """Replace `astroquery.gaia.Gaia` with a stub that records the query.

    Returns `(calls, table)`. The two stars sit about 1.5 arcsec apart around
    RA 10.05 deg, Dec -4.95 deg, and carry zero proper motion so that
    `StarTables.get_stars` can propagate them to any observation date.
    """
    calls = []
    table = Table(
        {
            "SOURCE_ID": [1, 2],
            "ra": [10.05, 10.0503] * u.deg,
            "dec": [-4.95, -4.9503] * u.deg,
            "pmra": [0.0, 0.0] * u.mas / u.yr,
            "pmdec": [0.0, 0.0] * u.mas / u.yr,
        }
    )

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
