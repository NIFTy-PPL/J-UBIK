import numpy as np
import pytest
from numpy.testing import assert_array_equal

from jubik.instruments.resolve.data import ms_import


class _Table:
    def __init__(self, columns):
        self._columns = columns

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getcol(self, name, startrow=0, nrow=-1):
        values = self._columns[name]
        if nrow == -1:
            return values[startrow:]
        return values[startrow : startrow + nrow]

    def nrows(self):
        return len(next(iter(self._columns.values())))


def test_data_description_maps_spectral_window_and_polarization(monkeypatch):
    table = _Table(
        {
            "SPECTRAL_WINDOW_ID": np.array([3, 1, 3, 0]),
            "POLARIZATION_ID": np.array([7, 5, 7, 9]),
        }
    )
    monkeypatch.setattr(ms_import, "ms_table", lambda path: table)

    assert_array_equal(ms_import._data_description_ids("mock.ms", 3), [0, 2])
    assert ms_import._pol_id("mock.ms", 3) == 7


def test_pol_id_rejects_multiple_setups_for_one_spectral_window(monkeypatch):
    table = _Table(
        {
            "SPECTRAL_WINDOW_ID": np.array([3, 3]),
            "POLARIZATION_ID": np.array([7, 8]),
        }
    )
    monkeypatch.setattr(ms_import, "ms_table", lambda path: table)

    with pytest.raises(ValueError, match="multiple polarization setups"):
        ms_import._pol_id("mock.ms", 3)


def test_first_pass_filters_by_mapped_data_description_ids(monkeypatch):
    table = _Table(
        {
            "FLAG": np.zeros((4, 1, 1), dtype=bool),
            "WEIGHT_SPECTRUM": np.ones((4, 1, 1), dtype=np.float32),
            "FIELD_ID": np.zeros(4, dtype=int),
            "DATA_DESC_ID": np.array([2, 3, 4, 1]),
        }
    )
    monkeypatch.setattr(ms_import, "ms_table", lambda path: table)
    monkeypatch.setattr(
        ms_import, "_determine_weighting", lambda path: (True, "WEIGHT_SPECTRUM")
    )
    monkeypatch.setattr(ms_import, "_ms_nchannels", lambda path, spw: 1)
    monkeypatch.setattr(
        ms_import, "_data_description_ids", lambda path, spw: np.array([2, 4])
    )

    active_rows, active_channels = ms_import._first_pass(
        "mock.ms",
        field=0,
        spectral_window=3,
        channels=slice(None),
        pol_indices=slice(None),
        pol_summation=False,
        ignore_flags=False,
    )

    assert_array_equal(active_rows, [True, False, True, False])
    assert_array_equal(active_channels, [True])
