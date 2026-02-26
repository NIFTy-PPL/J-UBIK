from pathlib import Path

import numpy as np

import jubik as ju


def test_pickle_roundtrip(tmp_path):
    payload = {"a": np.array([1, 2, 3]), "b": {"x": 1.5}}
    path = tmp_path / "payload.pkl"

    ju.save_to_pickle(payload, path)
    loaded = ju.load_from_pickle(path)

    assert loaded["b"] == payload["b"]
    np.testing.assert_array_equal(loaded["a"], payload["a"])


def test_create_output_directory_is_idempotent(tmp_path):
    outdir = tmp_path / "nested" / "dir"

    first = ju.create_output_directory(str(outdir))
    second = ju.create_output_directory(str(outdir))

    assert Path(first) == outdir
    assert Path(second) == outdir
    assert outdir.is_dir()


def test_add_functions_returns_sum():
    f = lambda x: x + 1
    g = lambda x: 2 * x
    h = ju.add_functions(f, g)

    assert h(3) == 10
    assert h(-2) == -5


def test_get_stats_matches_known_mean_and_std():
    sample_list = [
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
    ]

    mean, std = ju.get_stats(sample_list, lambda x: x)

    np.testing.assert_allclose(mean, np.array([1.0, 2.0]))
    np.testing.assert_allclose(std, np.array([1.0, 1.0]))


def test_save_to_yaml_and_get_config_roundtrip(tmp_path):
    cfg = {
        "alpha": 1,
        "beta": [1, 2, 3],
        "nested": {"flag": True, "value": 0.5},
    }

    ju.save_to_yaml(cfg, "config.yaml", dir=str(tmp_path), verbose=False)
    loaded = ju.get_config(str(tmp_path / "config.yaml"))

    assert loaded == cfg


def test_coord_center_small_grid():
    res = ju.coord_center(side_length=4, side_n=2)
    np.testing.assert_array_equal(res, np.array([5, 13, 7, 15]))
