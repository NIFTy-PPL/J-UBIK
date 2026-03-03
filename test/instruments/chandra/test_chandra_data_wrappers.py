from os.path import join

import jax.numpy as jnp
import nifty.re as jft
import numpy as np

import jubik.instruments.chandra.chandra_data as cd


def _config(tmp_path, mock_gen_config=None):
    return {
        "telescope": {
            "fov": 30.0,
            "center_obs_id": "center",
        },
        "files": {
            "res_dir": str(tmp_path),
            "data_dict": "masked.pkl",
            "processed_obs_folder": "processed",
            "mock_gen_config": mock_gen_config,
        },
        "grid": {
            "sdim": 3,
            "edim": 2,
            "energy_bin": {
                "e_min": [0.5, 1.0],
                "e_max": [1.0, 2.0],
            },
        },
        "plotting": {"enabled": False},
        "obs_info": {
            "other": {"tag": "other"},
            "center": {"tag": "center"},
        },
        "seed": 17,
    }


def test_generate_chandra_data_loads_cached_data(tmp_path, monkeypatch):
    file_info = {
        "res_dir": str(tmp_path),
        "processed_obs_folder": "processed",
    }
    tel_info = {"fov": 30.0}
    grid_info = {"sdim": 3, "edim": 2, "energy_bin": {"e_min": [0.5], "e_max": [1.0]}}
    obs_info = {"obs": {"tag": "obs"}}

    outroot = str(tmp_path / "processed")
    cached = jnp.ones((1, 2, 3, 3), dtype=int)

    monkeypatch.setattr(cd, "create_output_directory", lambda _: outroot)
    monkeypatch.setattr(cd, "exists", lambda path: path == join(outroot, "data.pkl"))
    monkeypatch.setattr(cd, "load_from_pickle", lambda _: cached)

    class ShouldNotBeCalled:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ChandraObservationInformation should not be called for cached data")

    monkeypatch.setattr(cd, "ChandraObservationInformation", ShouldNotBeCalled)

    result = cd.generate_chandra_data(file_info, tel_info, grid_info, obs_info)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(cached))


def test_generate_chandra_data_reorders_center_obs_and_passes_center(tmp_path, monkeypatch):
    file_info = {
        "res_dir": str(tmp_path),
        "processed_obs_folder": "processed",
    }
    tel_info = {"fov": 30.0, "center_obs_id": "center"}
    grid_info = {"sdim": 3, "edim": 2, "energy_bin": {"e_min": [0.5], "e_max": [1.0]}}
    obs_info = {
        "other": {"tag": "other"},
        "center": {"tag": "center"},
    }

    monkeypatch.setattr(cd, "create_output_directory", lambda _: str(tmp_path / "processed"))
    monkeypatch.setattr(cd, "exists", lambda _: False)

    class FakeInfo:
        calls = []
        centers = []

        def __init__(self, obs, npix_s, npix_e, fov, elim, energy_ranges=None, center=None):
            FakeInfo.calls.append(obs["tag"])
            FakeInfo.centers.append(center)
            self._obs = obs
            self.obsInfo = {"aim_ra": 111.0, "aim_dec": -5.0}

        def get_data(self, _):
            base = 100 if self._obs["tag"] == "center" else 10
            return np.arange(3 * 3 * 2).reshape((3, 3, 2)) + base

    monkeypatch.setattr(cd, "ChandraObservationInformation", FakeInfo)

    result = cd.generate_chandra_data(file_info, tel_info, grid_info, obs_info)

    assert FakeInfo.calls == ["center", "other"]
    assert FakeInfo.centers[0] is None
    assert FakeInfo.centers[1] == (111.0, -5.0)
    assert result.shape == (2, 2, 3, 3)

    center_expected = np.transpose(np.arange(3 * 3 * 2).reshape((3, 3, 2)) + 100)
    np.testing.assert_array_equal(np.asarray(result[0]), center_expected)


def test_create_chandra_data_from_config_mock_generation_path(tmp_path, monkeypatch):
    config = _config(tmp_path, mock_gen_config="mock_config.yaml")
    response_dict = {"mask": lambda x: x}

    calls = {"create_mock_data": None, "copy_config": None}

    monkeypatch.setattr(cd, "exists", lambda _: False)
    monkeypatch.setattr(cd, "get_config", lambda _: {"priors": {"k": 1}})
    monkeypatch.setattr(
        cd,
        "create_mock_data",
        lambda tel, files, grid, priors, plot, seed, resp: calls.update(
            {"create_mock_data": (tel, files, grid, priors, plot, seed, resp)}
        ),
    )
    monkeypatch.setattr(
        cd, "copy_config", lambda src, output_dir=None: calls.update({"copy_config": (src, output_dir)})
    )

    def _should_not_call(*args, **kwargs):
        raise AssertionError("generate_chandra_data should not be called in mock path")

    monkeypatch.setattr(cd, "generate_chandra_data", _should_not_call)

    cd.create_chandra_data_from_config(config, response_dict)

    assert config["telescope"]["tm_ids"] == list(config["obs_info"].keys())
    assert calls["create_mock_data"] is not None
    assert calls["create_mock_data"][3] == {"k": 1}
    assert calls["copy_config"] == ("mock_config.yaml", str(tmp_path))


def test_create_chandra_data_from_config_real_generation_path(tmp_path, monkeypatch):
    config = _config(tmp_path, mock_gen_config=None)
    generated = np.arange(2 * 2 * 3 * 3).reshape((2, 2, 3, 3))
    saved = {}

    monkeypatch.setattr(cd, "exists", lambda _: False)
    monkeypatch.setattr(cd, "generate_chandra_data", lambda *args, **kwargs: generated)
    monkeypatch.setattr(
        cd, "save_to_pickle", lambda obj, path: saved.update({"obj": obj, "path": path})
    )
    response_dict = {"mask": lambda x: jft.Vector({"tm_1": x})}

    cd.create_chandra_data_from_config(config, response_dict)

    assert config["telescope"]["tm_ids"] == list(config["obs_info"].keys())
    assert "obj" in saved and "path" in saved
    assert saved["path"] == join(str(tmp_path), "masked.pkl")
    assert set(saved["obj"].keys()) == {"tm_1"}


def test_create_chandra_data_from_config_skips_when_data_exists(tmp_path, monkeypatch):
    config = _config(tmp_path, mock_gen_config=None)
    response_dict = {"mask": lambda x: x}
    data_path = join(config["files"]["res_dir"], config["files"]["data_dict"])

    monkeypatch.setattr(cd, "exists", lambda path: path == data_path)

    def _should_not_call(*args, **kwargs):
        raise AssertionError("No generation should happen when data already exists")

    monkeypatch.setattr(cd, "generate_chandra_data", _should_not_call)
    monkeypatch.setattr(cd, "create_mock_data", _should_not_call)
    monkeypatch.setattr(cd, "save_to_pickle", _should_not_call)

    cd.create_chandra_data_from_config(config, response_dict)
