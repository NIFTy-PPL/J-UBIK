import json

import jax
import jax.numpy as jnp
import nifty.re as jft
import pytest

from jubik.profiling import (ProfilingCallback, named_models_from_lens_system,
                             profile_model, profile_tree)

jax.config.update('jax_platform_name', 'cpu')


@pytest.fixture
def sub_models():
    diffuse = jft.Model(
        lambda x: jnp.fft.fft2(x['xi']).real ** 2,
        domain={'xi': jft.ShapeWithDtype((16, 16), jnp.float64)})
    points = jft.Model(
        lambda x: jnp.exp(x['points']),
        domain={'points': jft.ShapeWithDtype((16, 16), jnp.float64)})
    return diffuse, points


def test_profile_model_jft_model(sub_models):
    diffuse, _ = sub_models
    row = profile_model(diffuse, name='diffuse', n=3)
    assert row.name == 'diffuse'
    assert row.n_params == 16 * 16
    assert row.compile_s > 0
    assert row.runtime_s > 0
    assert row.grad_runtime_s is None


def test_profile_model_grad(sub_models):
    diffuse, _ = sub_models
    row = profile_model(diffuse, grad=True, n=3)
    assert row.grad_compile_s > 0
    assert row.grad_runtime_s > 0


def test_profile_model_plain_callable_needs_input():
    with pytest.raises(ValueError, match='domain'):
        profile_model(lambda x: x ** 2)
    row = profile_model(lambda x: x ** 2, x=jnp.ones((8, 8)), name='sq', n=3)
    assert row.runtime_s > 0
    assert row.n_params is None


def test_profile_tree_with_root_and_json(sub_models, tmp_path):
    diffuse, points = sub_models
    root = jft.Model(lambda x: diffuse(x) + points(x),
                     domain=diffuse.domain | points.domain)
    report = profile_tree({'diffuse': diffuse, 'points': points}, root=root,
                          n=3, verbose=False)
    assert [r.name for r in report.rows] == ['diffuse', 'points']
    assert report.root.name == 'TOTAL (fused)'
    assert report.fusion_gap() > 0

    table = str(report)
    assert 'diffuse' in table and 'TOTAL (fused)' in table

    out = tmp_path / 'profile.json'
    report.to_json(out)
    payload = json.loads(out.read_text())
    assert len(payload['rows']) == 2
    assert payload['root']['name'] == 'TOTAL (fused)'


def test_named_models_from_lens_system_duck_typing(sub_models):
    diffuse, points = sub_models

    class FakeSub:
        def __init__(self, model):
            self.model = model

    class FakeSystem:
        _slots = {'lens.light': FakeSub(diffuse),
                  'source.light': FakeSub(points),
                  'lens.deflection': FakeSub(None)}

        def paths(self):
            return list(self._slots)

        def __getitem__(self, path):
            return self._slots[path]

    named = named_models_from_lens_system(FakeSystem())
    assert set(named) == {'lens.light', 'source.light'}


def test_profiling_callback_writes_jsonl(sub_models, tmp_path):
    class FakeState:
        def __init__(self, nit):
            self.nit = nit

    path = tmp_path / 'iterations.jsonl'
    callback = ProfilingCallback(path=path)
    callback(None, FakeState(1))
    callback(None, FakeState(2))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['nit'] for r in records] == [1, 2]
    assert records[0]['wall_s'] is None
    assert records[1]['wall_s'] > 0
