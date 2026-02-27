from os.path import join

from jax import random
import nifty.re as jft
import numpy as np
import pytest

import jubik.sugar_plot as sp


class _EmptySamples:
    def __init__(self, pos):
        self.pos = pos

    def __len__(self):
        return 0


class _SamplesContainer:
    def __init__(self, samples):
        self.samples = samples


def test_plot_pspec_uses_pos_when_sample_list_is_empty(tmp_path, monkeypatch):
    import nifty.re.correlated_field as cf

    monkeypatch.setattr(
        cf,
        "get_fourier_mode_distributor",
        lambda shape, distances: (None, np.array([1.0, 2.0, 3.0]), None),
    )

    sample_list = _EmptySamples(pos={"dummy": 1})
    pspec = lambda _: np.array([1.0, 2.0, 3.0])

    sp.plot_pspec(
        pspec=pspec,
        shape=(4, 4),
        distances=1.0,
        sample_list=sample_list,
        output_directory=str(tmp_path),
        iteration=0,
    )

    out = tmp_path / "spatial_pspec" / "samples_0.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_sample_and_stats_plots_samples_and_stats(monkeypatch):
    plot_calls = []
    rgb_calls = []

    monkeypatch.setattr(sp, "create_output_directory", lambda p: p)
    monkeypatch.setattr(
        sp,
        "plot_result",
        lambda array, output_file=None, **kwargs: plot_calls.append((array, output_file, kwargs)),
    )
    monkeypatch.setattr(sp, "plot_rgb", lambda *args, **kwargs: rgb_calls.append(kwargs.get("name")))
    monkeypatch.setattr(
        sp,
        "get_stats",
        lambda sample_list, op: (np.ones((3, 2, 2)) * 5.0, np.ones((3, 2, 2)) * 2.0),
    )

    operators = {"sky": lambda s: np.ones((3, 2, 2)) * float(s), "none": None}
    plotting_kwargs = {"n_rows": 1, "n_cols": 1, "figsize": (2, 2), "title": "tmp"}

    sp.plot_sample_and_stats(
        output_directory="out",
        operators_dict=operators,
        sample_list=[1.0, 2.0],
        iteration=0,
        plotting_kwargs=plotting_kwargs,
        plot_samples=True,
        plot_rgb_samples=False,
    )

    assert len(plot_calls) == 4  # 2 sample images + mean + std
    assert any("sample_1_it_0.png" in c[1] for c in plot_calls)
    assert any("sample_2_it_0.png" in c[1] for c in plot_calls)
    assert any("mean_it_0.png" in c[1] for c in plot_calls)
    assert any("std_it_0.png" in c[1] for c in plot_calls)
    # `plot_rgb_samples=False` only disables RGB sample plots; RGB stats are
    # still produced for three-channel outputs.
    assert len(rgb_calls) == 2
    assert any(name.endswith("_mean_it_0_rgb") for name in rgb_calls)
    assert any(name.endswith("_mean_it_0_rgb_log") for name in rgb_calls)
    # The function strips layout keys before the statistics plots.
    assert "n_rows" not in plotting_kwargs
    assert "n_cols" not in plotting_kwargs
    assert "figsize" not in plotting_kwargs
    assert "title" not in plotting_kwargs


def test_plot_erosita_priors_without_signal_response(monkeypatch):
    calls = []

    class FakeSkyModel:
        class _Op:
            def __init__(self, offset):
                self.domain = {"dummy": None}
                self._offset = offset

            def __call__(self, pos):
                return np.ones((2, 2, 2)) * (float(pos) + self._offset)

        def __init__(self, config):
            self.config = config

        def create_sky_model(self):
            return None

        def sky_model_to_dict(self):
            return {
                "sky": self._Op(0.0),
                "diffuse": self._Op(1.0),
            }

    monkeypatch.setattr(sp, "SkyModel", FakeSkyModel)
    monkeypatch.setattr(sp.jft, "random_like", lambda key, domain: 2.0)
    monkeypatch.setattr(sp, "create_output_directory", lambda p: p)
    monkeypatch.setattr(sp, "plot_result", lambda arr, output_file=None, **kwargs: calls.append(output_file))

    cfg = {
        "grid": {"energy_bin": {"e_min": [0.5, 1.0], "e_max": [1.0, 2.0]}},
        "telescope": {"tm_ids": [1, 2]},
    }
    sp.plot_erosita_priors(
        key=random.PRNGKey(0),
        n_samples=2,
        config=cfg,
        priors_dir="priors",
        signal_response=False,
        plotting_kwargs={},
    )

    assert len(calls) == 4  # 2 samples * 2 operators
    assert any("sample_0/priors_sky.png" in c for c in calls)
    assert any("sample_1/priors_diffuse.png" in c for c in calls)


def test_plot_uncertainty_weighted_residuals_defaults_and_hist(monkeypatch):
    monkeypatch.setattr(
        sp,
        "calculate_uwr",
        lambda *args, **kwargs: (
            np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]),
            np.array([[True, False, True], [True, False, True]]),
        ),
    )
    plot_calls = []
    hist_calls = []
    monkeypatch.setattr(
        sp,
        "plot_result",
        lambda arr, output_file=None, **kwargs: plot_calls.append((arr, output_file, kwargs)),
    )
    monkeypatch.setattr(
        sp,
        "plot_histograms",
        lambda hist, edges, filename, **kwargs: hist_calls.append((hist, edges, filename, kwargs)),
    )

    samples = _SamplesContainer(samples=np.array([0.0]))
    operator_dict = {"sky": lambda x: x, "pspec": lambda x: x}
    reference_dict = {"sky": np.ones((2, 3))}

    res = sp.plot_uncertainty_weighted_residuals(
        samples=samples,
        operator_dict=operator_dict,
        diagnostics_path="diag",
        response_dict={},
        reference_dict=reference_dict,
        base_filename="uwr_",
        mask=True,
        n_bins=5,
        range=(-3, 3),
        plot_kwargs={},
    )

    assert set(res.keys()) == {"sky"}
    assert np.isnan(res["sky"]["masked_uwrs"][:, 1]).all()
    assert len(plot_calls) == 1
    assert plot_calls[0][1] == join("diag", "uwr_sky.png")
    assert plot_calls[0][2]["cmap"] == "RdYlBu_r"
    assert plot_calls[0][2]["vmin"] == -5
    assert plot_calls[0][2]["vmax"] == 5
    assert len(hist_calls) == 1
    assert hist_calls[0][0].shape == (5,)


def test_plot_noise_weighted_residuals_creates_tm_plots_and_hist(monkeypatch):
    nwrs = np.arange(2 * 2 * 3 * 3, dtype=float).reshape((2, 2, 3, 3)) + 1.0
    mask = np.zeros_like(nwrs, dtype=bool)
    mask[0, 0, 0, 0] = True

    monkeypatch.setattr(sp, "calculate_nwr", lambda *args, **kwargs: (nwrs.copy(), mask.copy()))
    monkeypatch.setattr(sp, "create_output_directory", lambda p: p)

    plot_calls = []
    hist_calls = []
    monkeypatch.setattr(sp, "plot_result", lambda arr, output_file=None, **kwargs: plot_calls.append(output_file))
    monkeypatch.setattr(sp, "plot_histograms", lambda hist, edges, filename, **kwargs: hist_calls.append(filename))

    samples = _SamplesContainer(samples=np.array([0.0]))
    operator_dict = {"sky": lambda x: x, "pspec": lambda x: x}

    res = sp.plot_noise_weighted_residuals(
        samples=samples,
        operator_dict=operator_dict,
        diagnostics_path="diag",
        response_dict={},
        reference_data=np.zeros_like(nwrs),
        base_filename="nwr_",
        n_bins=4,
        extent=(-2, 2),
        plot_kwargs={},
    )

    assert set(res.keys()) == {"sky"}
    assert np.isnan(res["sky"]["masked_nwrs"][0, 0, 0, 0])
    assert len(plot_calls) == 6  # 2 TMs * (2 samples + mean)
    assert len(hist_calls) == 2  # one histogram per TM


def test_plot_2d_gt_vs_rec_histogram_single_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sp,
        "plot_sample_averaged_log_2d_histogram",
        lambda **kwargs: calls.append(kwargs),
    )

    samples = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    operator_dict = {"sky": lambda s: np.asarray(s), "pspec": lambda s: s}
    response_dict = {"R": lambda x: jft.Vector({"tm": np.asarray(x)})}
    reference_dict = {"sky": np.array([1.0, 2.0])}

    sp.plot_2d_gt_vs_rec_histogram(
        samples=samples,
        operator_dict=operator_dict,
        diagnostics_path="diag",
        response_dict=response_dict,
        reference_dict=reference_dict,
        base_filename="cmp_",
        response=True,
        type="single",
        plot_kwargs={"x_label": "gt", "y_label": "rec"},
    )

    assert len(calls) == 1
    assert calls[0]["output_path"] == join("diag", "cmp_hist_sky.png")
    assert len(calls[0]["x_array_list"]) == 1
    assert len(calls[0]["y_array_list"]) == 1


def test_plot_2d_gt_vs_rec_histogram_invalid_type_raises():
    samples = [np.array([1.0, 2.0])]
    operator_dict = {"sky": lambda s: np.asarray(s)}
    response_dict = {"R": lambda x: jft.Vector({"tm": np.asarray(x)})}
    reference_dict = {"sky": np.array([1.0, 2.0])}

    with pytest.raises(NotImplementedError):
        sp.plot_2d_gt_vs_rec_histogram(
            samples=samples,
            operator_dict=operator_dict,
            diagnostics_path="diag",
            response_dict=response_dict,
            reference_dict=reference_dict,
            base_filename="cmp_",
            response=True,
            type="invalid",
            plot_kwargs={"x_label": "gt", "y_label": "rec"},
        )
