import nifty.re as jft
import numpy as np
import pytest

import jubik as ju


def test_load_masked_data_from_config_success(tmp_path):
    res_dir = tmp_path / "results"
    data_name = "masked.pkl"
    payload = {"tm_1": np.array([1, 0, 3], dtype=int)}

    ju.create_output_directory(str(res_dir))
    ju.save_to_pickle(payload, res_dir / data_name)

    cfg = {"files": {"res_dir": str(res_dir), "data_dict": data_name}}
    loaded = ju.load_masked_data_from_config(cfg)

    assert isinstance(loaded, jft.Vector)
    assert set(loaded.tree.keys()) == {"tm_1"}
    np.testing.assert_array_equal(np.asarray(loaded.tree["tm_1"]), payload["tm_1"])


def test_load_masked_data_from_config_missing_path_raises(tmp_path):
    cfg = {"files": {"res_dir": str(tmp_path), "data_dict": "missing.pkl"}}

    with pytest.raises(ValueError, match="Data path does not exist"):
        ju.load_masked_data_from_config(cfg)


def test_load_mock_position_from_config_success(tmp_path):
    res_dir = tmp_path / "results"
    pos_name = "pos.pkl"
    payload = {"latent": np.array([0.1, -0.2, 0.3])}

    ju.create_output_directory(str(res_dir))
    ju.save_to_pickle(payload, res_dir / pos_name)

    cfg = {"files": {"res_dir": str(res_dir), "pos_dict": pos_name}}
    loaded = ju.load_mock_position_from_config(cfg)

    assert set(loaded.keys()) == {"latent"}
    np.testing.assert_array_equal(np.asarray(loaded["latent"]), payload["latent"])


def test_load_mock_position_from_config_missing_path_raises(tmp_path):
    cfg = {"files": {"res_dir": str(tmp_path), "pos_dict": "missing.pkl"}}

    with pytest.raises(ValueError, match="Mock position path does not exist"):
        ju.load_mock_position_from_config(cfg)
