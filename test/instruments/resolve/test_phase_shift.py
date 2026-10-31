from os.path import realpath

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.likelihood.mosaic_likelihood import (
    build_likelihood_from_sky_beamer,
)
from jubik.instruments.resolve.data import load_and_modify_data_from_objects
from jubik.instruments.resolve.mosaicing.sky_beamer import BeamPattern, SkyBeamer
from jubik.instruments.resolve.parse.data import DataLoading
from jubik.instruments.resolve.parse.data.data_modify.pipeline import ObservationModify
from jubik.instruments.resolve.parse.phase_shift import PhaseShiftConfig
from jubik.instruments.resolve.phase_shift import (
    assign_shift_groups,
    build_phase_shift,
)
from jubik.instruments.resolve.response import _hermitian_adjoint

from generate_test_obs import generate_random_obs
from test_response import setup_grid_obs_response

jax.config.update("jax_enable_x64", True)

FREQS = np.array([1e9, 1.3e9])
OFFSET = ["normal", 0.0, 50.0]
PRIOR = ["normal", 0.0, 0.05]
EB1 = {"prior": ["normal", 0.0, 20.0], "data": ["a/a_spw{spw}.npz"]}
UNIT = ["normal", 0.0, 1.0]  # the shift is the latent itself


def phase_shift(obs, offset=UNIT):
    """The phase shift of `obs` for an offset alone, domain key `shift_offset`."""
    return build_phase_shift(
        PhaseShiftConfig.from_yaml_dict({"offset": offset}), None, obs
    )


def point_source_sky(grid, i, j):
    sky = np.zeros(grid.shape)
    sky[..., i, j] = 1.0
    return sky


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_shift_moves_the_sampling_positions(backend):
    """A shift of (a, b) pixels along the sky grid (row, column) moves the
    sampling positions of the model sky by (a, b), so a source at pixel
    (i, j) gives the visibilities of a source at (i - a, j - b) without shift.
    """
    np.random.seed(42)
    grid, obs, r_ducc, r_finufft = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    response = r_ducc if backend == "ducc" else r_finufft
    a, b, i, j = 3, -2, 40, 55
    pix_row, pix_col = grid.spatial.pixel_scales_yx.to_value(u.arcsec)

    phase = phase_shift(obs)({"shift_offset": np.array([a * pix_row, b * pix_col])})

    assert_allclose(
        phase * response(point_source_sky(grid, i, j)),
        response(point_source_sky(grid, i - a, j - b)),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "vis_dtype,weight_dtype", ((np.complex64, np.float32), (np.complex128, np.float64))
)
def test_zero_shift_is_identity(vis_dtype, weight_dtype):
    np.random.seed(43)
    pol = ju.polarization.PolarizationType(("LL", "RR"))
    obs = generate_random_obs(FREQS, 10, [-1e2, 1e2], [-5, 5], pol)
    obs = rve.Observation(
        obs.antenna_positions,
        obs.vis_val.astype(vis_dtype),
        obs.weight_val.astype(weight_dtype),
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    shift = phase_shift(obs, PRIOR)

    phase = shift(jft.zeros_like(shift.domain))

    assert phase.dtype == vis_dtype
    assert phase.shape == (1,) + obs.vis.shape[1:]
    assert_allclose(phase, 1.0)
    assert build_phase_shift(None, None, obs) is None


def data_cfg(templates, windows):
    return {
        "data_path": "/data",
        "data_templates": templates,
        "spectral": {"window": windows},
    }


def data_files(*data_cfgs):
    """The `DataFile` of every file of the data blocks `data_cfgs`."""
    return [
        file for cfg in data_cfgs for file in DataLoading.from_yaml_dict(cfg).data_files
    ]


def group_of(groups, *data_cfgs):
    """The name of the shift group of every file of `data_cfgs` by path, for
    the shift groups `groups` and an offset."""
    config = PhaseShiftConfig.from_yaml_dict({"offset": OFFSET} | groups)
    assigned = assign_shift_groups(config, data_files(*data_cfgs))
    return {path: g and g.name for path, g in assigned.items()}


def test_template_covers_every_spw_and_blocks_share_a_group():
    groups = {
        "eb1": {"prior": PRIOR, "data": ["a/a_spw{spw}.npz"]},
        "eb2_usb": {"prior": PRIOR, "data": ["b/b_spw17.npz"]},
    }
    templates = ["a/a_spw{spw}.npz", "b/b_spw{spw}.npz", "c/c_spw{spw}.npz"]

    assigned = group_of(groups, data_cfg(templates, [5, 7]), data_cfg(templates, [17]))

    assert assigned == {
        "/data/a/a_spw5.npz": "eb1",
        "/data/a/a_spw7.npz": "eb1",
        "/data/a/a_spw17.npz": "eb1",
        "/data/b/b_spw5.npz": None,
        "/data/b/b_spw7.npz": None,
        "/data/b/b_spw17.npz": "eb2_usb",
        "/data/c/c_spw5.npz": None,
        "/data/c/c_spw7.npz": None,
        "/data/c/c_spw17.npz": None,
    }


def test_file_in_two_groups_raises():
    groups = {
        "eb1": {"prior": PRIOR, "data": ["a/a_spw{spw}.npz"]},
        "eb1_usb": {"prior": PRIOR, "data": ["a/a_spw17.npz"]},
    }
    with pytest.raises(ValueError, match="a_spw17.npz"):
        group_of(groups, data_cfg(["a/a_spw{spw}.npz"], [5, 17]))


def test_file_of_two_blocks_in_two_groups_raises():
    """Two templates that expand to the same file must not give it two shifts."""
    groups = {
        "one": {"prior": PRIOR, "data": ["a{spw}.npz"]},
        "two": {"prior": PRIOR, "data": ["a{field}.npz"]},
    }
    other = data_cfg(["a{field}.npz"], [None])
    other["field_ids"] = [5]
    with pytest.raises(ValueError, match="a5.npz"):
        group_of(groups, data_cfg(["a{spw}.npz"], [5]), other)


def test_file_of_two_blocks_has_its_group_in_both():
    """A file listed through one block's template is shifted in the other too."""
    groups = {"one": {"prior": PRIOR, "data": ["a{spw}.npz"]}}
    other = data_cfg(["a{field}.npz"], [None])
    other["field_ids"] = [5]
    a, b = data_files(data_cfg(["a{spw}.npz"], [5]), other)
    assert a.path == b.path and a.template != b.template
    assert group_of(groups, data_cfg(["a{spw}.npz"], [5]), other) == {
        "/data/a5.npz": "one"
    }


def test_entry_is_relative_to_the_data_path_of_its_block():
    """The same entry names a different file in a block with another data_path."""
    groups = {"eb1": {"prior": PRIOR, "data": ["a/a_spw5.npz"]}}
    other = data_cfg(["a/a_spw{spw}.npz"], [5])
    other["data_path"] = "/other"
    assigned = group_of(groups, data_cfg(["a/a_spw{spw}.npz"], [5]), other)
    assert assigned == {"/data/a/a_spw5.npz": "eb1", "/other/a/a_spw5.npz": "eb1"}


@pytest.mark.parametrize("data", ["a_spw5.npz", [], ["a_spw5.npz", 5]])
def test_group_data_must_be_a_list_of_names(data):
    with pytest.raises(ValueError, match="eb1"):
        PhaseShiftConfig.from_yaml_dict(
            {"offset": OFFSET, "eb1": {"prior": PRIOR, "data": data}}
        )


@pytest.mark.parametrize("bad", ["offset", "group"])
def test_broken_prior_raises_when_parsed(bad):
    shift_cfg = {"offset": OFFSET, "eb1": {"prior": PRIOR, "data": ["a.npz"]}}
    if bad == "offset":
        shift_cfg["offset"] = ["normal", 0.0]
    else:
        shift_cfg["eb1"]["prior"] = ["normal", 0.0]
    with pytest.raises(TypeError):
        PhaseShiftConfig.from_yaml_dict(shift_cfg)


def test_entry_of_an_unloaded_block_is_ignored():
    groups = {"eb1": {"prior": PRIOR, "data": ["a/a_spw9.npz"]}}
    assigned = group_of(groups, data_cfg(["a/a_spw{spw}.npz"], [5]))
    assert assigned == {"/data/a/a_spw5.npz": None}


def test_equivalent_paths_match():
    groups = {"eb1": {"prior": PRIOR, "data": ["a/a_spw5.npz"]}}
    cfg = data_cfg(["./a/a_spw{spw}.npz"], [5])
    cfg["data_path"] = "/data/"
    assert group_of(groups, cfg) == {"/data/a/a_spw5.npz": "eb1"}


@pytest.mark.parametrize("entry", ["/data/eb/a_spw5.npz", "../eb/a_spw{spw}.npz"])
def test_entry_is_joined_to_the_data_path(entry):
    """An absolute entry, or one leaving and re-entering data_path, still matches."""
    groups = {"eb1": {"prior": PRIOR, "data": [entry]}}
    cfg = data_cfg(["../eb/a_spw{spw}.npz"], [5])
    cfg["data_path"] = "/data/eb"
    assert group_of(groups, cfg) == {"/data/eb/a_spw5.npz": "eb1"}


def test_data_file_resolves_symlinks_before_parent_dirs(tmp_path):
    """`..` after a symlinked data_path leaves the link target, as the OS does."""
    (tmp_path / "remote" / "child").mkdir(parents=True)
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "link").symlink_to(tmp_path / "remote" / "child")
    cfg = data_cfg(["../obs.npz"], [None])
    cfg["data_path"] = str(tmp_path / "local" / "link")

    (file,) = data_files(cfg)

    assert file.path == realpath(tmp_path / "remote" / "obs.npz")
    assert file.matches("../obs.npz")


def test_loader_yields_each_observation_with_its_file(tmp_path):
    """The shift reaches an observation through its own file, not its position."""
    tmp_path = tmp_path.resolve()  # `DataFile.path` is canonical
    pol = ju.polarization.PolarizationType(("LL", "RR"))
    cfg = {
        "data_path": str(tmp_path),
        "data_templates": ["a_spw{spw}.npz", "c_spw{spw}.npz"],
        "spectral": {"window": [5, 7]},
    }
    files = data_files(cfg)
    saved = {}
    for seed, file in enumerate(files):
        np.random.seed(seed)
        saved[file.path] = generate_random_obs(FREQS, 10, [-1e2, 1e2], [-5, 5], pol)
        saved[file.path].save(file.path, False)
    config = PhaseShiftConfig.from_yaml_dict(
        {"offset": OFFSET, "a": {"prior": OFFSET, "data": ["a_spw7.npz"]}}
    )
    groups = assign_shift_groups(config, files)

    loaded = load_and_modify_data_from_objects(
        ju.Color([0.5e9, 2e9] * u.Hz),
        DataLoading.from_yaml_dict(cfg),
        ObservationModify.from_yaml_dict({}),
    )
    seen = {}
    for file, obs in loaded:
        assert_allclose(obs.vis_val, saved[file.path].vis_val)
        shift = build_phase_shift(config, groups[file.path], obs)
        seen[file.path] = shift.domain.keys()

    assert seen == {
        str(tmp_path / "a_spw5.npz"): {"shift_offset"},
        str(tmp_path / "a_spw7.npz"): {"shift_offset", "shift_a"},
        str(tmp_path / "c_spw5.npz"): {"shift_offset"},
        str(tmp_path / "c_spw7.npz"): {"shift_offset"},
    }


def file_shifts(shift_cfg, obs):
    """The phase shift of `obs` as files a (in `eb1` if set) and c of one block."""
    config = PhaseShiftConfig.from_yaml_dict(shift_cfg)
    files = data_files(data_cfg(["a/a_spw{spw}.npz", "c/c_spw{spw}.npz"], [5]))
    groups = assign_shift_groups(config, files)
    return [build_phase_shift(config, groups[f.path], obs) for f in files]


def fixed_shift_phase(obs, shift):
    """The phase of `obs` for the shift `shift` in arcsec."""
    return phase_shift(obs)({"shift_offset": shift})


def offset_obs():
    np.random.seed(46)
    pol = ju.polarization.PolarizationType(("LL", "RR"))
    return generate_random_obs(FREQS, 10, [-1e2, 1e2], [-5, 5], pol)


def test_offset_is_the_prior_itself():
    config = PhaseShiftConfig.from_yaml_dict({"offset": OFFSET, "eb1": EB1})
    assert [g.name for g in config.groups] == ["eb1"]
    assert config.groups[0].data == ("a/a_spw{spw}.npz",)
    as_dict = {"distribution": "normal", "mean": 0.0, "sigma": 50.0}
    for offset in (OFFSET, as_dict):
        prior = PhaseShiftConfig.from_yaml_dict({"offset": offset}).offset
        assert prior.distribution == "normal"
        assert_allclose(prior.mean, [0.0, 0.0])
        assert_allclose(prior.sigma, [50.0, 50.0])
    assert PhaseShiftConfig.from_yaml_dict(None) is None


@pytest.mark.parametrize(
    "offset", [{"prior": OFFSET}, {"prior": OFFSET, "data": ["a/a_spw5.npz"]}]
)
def test_offset_written_as_a_group_raises(offset):
    with pytest.raises(ValueError, match="prior itself"):
        PhaseShiftConfig.from_yaml_dict({"offset": offset})


@pytest.mark.parametrize("shift_cfg", [{"eb1": EB1}, {}, {"offset": None}])
def test_shift_needs_the_offset(shift_cfg):
    with pytest.raises(ValueError, match="offset"):
        file_shifts(shift_cfg, offset_obs())


def test_offset_alone_moves_every_file():
    obs = offset_obs()
    x = {"shift_offset": np.array([0.7, -1.3])}
    expected = fixed_shift_phase(obs, 50.0 * x["shift_offset"])
    assert not np.allclose(expected, 1.0)

    for shift in file_shifts({"offset": OFFSET}, obs):
        assert shift.domain.keys() == {"shift_offset"}
        assert_allclose(shift(x), expected)


def test_group_adds_to_the_offset_of_its_own_files():
    obs = offset_obs()
    x = {"shift_offset": np.array([0.7, -1.3]), "shift_eb1": np.array([-0.4, 0.9])}
    offset, eb1 = 50.0 * x["shift_offset"], 20.0 * x["shift_eb1"]

    in_group, in_none = file_shifts({"offset": OFFSET, "eb1": EB1}, obs)

    assert in_group.domain.keys() == {"shift_offset", "shift_eb1"}
    assert in_none.domain.keys() == {"shift_offset"}
    assert_allclose(in_group(x), fixed_shift_phase(obs, offset + eb1))
    assert_allclose(in_none(x), fixed_shift_phase(obs, offset))


def test_without_shift_no_file_is_shifted():
    assert file_shifts(None, offset_obs()) == [None, None]


def test_likelihood_has_the_shift_only_when_set():
    np.random.seed(44)
    grid, obs, _, _ = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape),
        {"band7_fld": BeamPattern(0.0, 0.0, np.ones(grid.shape), None)},
    )

    for prior in (None, ["normal", 0.0, 0.05]):
        shift = None if prior is None else phase_shift(obs, prior)
        likelihood = build_likelihood_from_sky_beamer(
            observation=obs,
            field_name="band7_fld",
            sky_beamer=sky_beamer,
            sky_grid=grid,
            backend_settings=rve.parse.FinufftSettings(epsilon=1e-9),
            shift=shift,
        ).likelihood
        domain = likelihood.domain.tree

        assert ("shift_offset" in domain) == (prior is not None)
        latent = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
        assert np.isfinite(likelihood.energy(latent))

    assert domain["shift_offset"].shape == (2,)


def build_band7(backend, prior):
    """Builder of one observation of band7 with one field, and a sky sample."""
    np.random.seed(45)
    grid, obs, _, _ = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape),
        {"band7_fld": BeamPattern(0.0, 0.0, np.ones(grid.shape), None)},
    )
    settings = {
        "ducc": rve.parse.Ducc0Settings(
            epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
        ),
        "finufft": rve.parse.FinufftSettings(epsilon=1e-9),
    }[backend]
    shift = None if prior is None else phase_shift(obs, prior)
    lhb = build_likelihood_from_sky_beamer(
        observation=obs,
        field_name="band7_fld",
        sky_beamer=sky_beamer,
        sky_grid=grid,
        backend_settings=settings,
        shift=shift,
    )
    return lhb, {"band7_fld": np.random.normal(size=grid.shape)}


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_response_adjoint_holds_the_shift_fixed(backend):
    """<R_x s, d> == <s, R_x^H d> with the shift fixed at a nonzero sample x."""
    # 60 arcsec std: phases of order one on the short test baselines
    lhb, sky = build_band7(backend, ["normal", 0.0, 60.0])
    unshifted, _ = build_band7(backend, None)
    x = {"shift_offset": np.array([0.7, -1.3])}
    d = lhb.visibilities

    adjoint = lhb.response_adjoint(d, x)["band7_fld"]

    lhs = np.real(np.sum(np.conj(lhb.response(sky | x)) * d))
    assert_allclose(lhs, np.sum(sky["band7_fld"] * np.real(adjoint)), rtol=1e-6)
    # The phase reaches the adjoint, so it differs from the unshifted one.
    plain = unshifted.response_adjoint(d)["band7_fld"]
    assert np.linalg.norm(adjoint - plain) > 0.1 * np.linalg.norm(plain)
    assert_allclose(lhb.response_adjoint(d, jft.Vector(x))["band7_fld"], adjoint)


def test_response_adjoint_needs_the_sample_with_a_shift():
    lhb, _ = build_band7("finufft", ["normal", 0.0, 1.0])
    with pytest.raises(ValueError, match="shift"):
        lhb.response_adjoint(lhb.visibilities)


def test_response_adjoint_without_shift_is_unchanged():
    lhb, _ = build_band7("finufft", None)
    d = lhb.visibilities
    expected = _hermitian_adjoint(lhb.response, lhb.response.domain, d)
    assert_allclose(
        lhb.response_adjoint(d, {"ignored": 0})["band7_fld"], expected["band7_fld"]
    )
