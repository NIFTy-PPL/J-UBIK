from os.path import join

import jax.numpy as jnp
import numpy as np

import jubik.instruments.chandra.chandra_likelihood as chl
import jubik.instruments.chandra.chandra_response as chrsp


def _response_config(tmp_path):
    return {
        "obs_info": {"obs_a": {"id": "a"}, "obs_b": {"id": "b"}},
        "grid": {
            "sdim": 3,
            "edim": 2,
            "energy_bin": {"e_min": [0.5, 1.0], "e_max": [1.0, 2.0]},
        },
        "files": {"res_dir": str(tmp_path), "processed_obs_folder": "processed"},
        "psf": {"npatch": 2, "margfrac": 0.1, "num_rays": 8},
        "telescope": {"fov": 30.0, "exp_cut": None},
    }


def test_build_chandra_response_from_config_uses_cached_exposure_and_psf(tmp_path, monkeypatch):
    config = _response_config(tmp_path)
    outroot = str(tmp_path / "processed")
    exposure_path = join(outroot, "exposure.pkl")
    psf_path = join(outroot, "psf.pkl")

    exposures = np.ones((2, 2, 3, 3))
    psfs = np.ones((2, 2, 3, 3))

    monkeypatch.setattr(chrsp, "create_output_directory", lambda _: outroot)
    monkeypatch.setattr(chrsp, "exists", lambda p: p in (exposure_path, psf_path))
    monkeypatch.setattr(chrsp, "load_from_pickle", lambda p: exposures if p == exposure_path else psfs)
    monkeypatch.setattr(chrsp, "linpatch_convolve", lambda x, domain, kernel, npatch, margin: x)

    class ShouldNotBeCalled:
        def __init__(self, *args, **kwargs):
            raise AssertionError("No observation object should be built when caches exist")

    def should_not_call(*args, **kwargs):
        raise AssertionError("get_psfpatches should not be called when psf cache exists")

    monkeypatch.setattr(chrsp, "ChandraObservationInformation", ShouldNotBeCalled)
    monkeypatch.setattr(chrsp, "get_psfpatches", should_not_call)

    response_dict = chrsp.build_chandra_response_from_config(config)

    assert set(response_dict.keys()) == {"pix_area", "psf", "exposure", "mask", "R"}
    assert response_dict["pix_area"] == (30.0 / 3) ** 2

    x = jnp.ones((2, 3, 3))
    masked = response_dict["R"](x)
    assert set(masked.tree.keys()) == {"obs_a", "obs_b"}
    for v in masked.tree.values():
        assert np.asarray(v).size == 2 * 3 * 3


def test_generate_chandra_likelihood_from_config_wires_build_create_load(monkeypatch):
    calls = {"build": 0, "create": 0, "load": 0, "amend": 0}
    response_func = lambda x: x

    monkeypatch.setattr(
        chl,
        "build_chandra_response_from_config",
        lambda cfg: calls.update({"build": calls["build"] + 1}) or {"R": response_func},
    )
    monkeypatch.setattr(
        chl,
        "create_chandra_data_from_config",
        lambda cfg, resp: calls.update({"create": calls["create"] + 1}),
    )
    monkeypatch.setattr(
        chl,
        "load_masked_data_from_config",
        lambda cfg: calls.update({"load": calls["load"] + 1}) or "masked-data",
    )

    class FakePoissonian:
        def __init__(self, data):
            self.data = data

        def amend(self, response):
            calls["amend"] += 1
            return {"data": self.data, "response": response}

    monkeypatch.setattr(chl.jft, "Poissonian", FakePoissonian)

    result = chl.generate_chandra_likelihood_from_config({"dummy": True})

    assert result == {"data": "masked-data", "response": response_func}
    assert calls == {"build": 1, "create": 1, "load": 1, "amend": 1}
