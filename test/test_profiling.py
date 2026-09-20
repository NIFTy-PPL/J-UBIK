import json

import jax
import jax.numpy as jnp
import nifty.re as jft
import pytest

from jubik.profiling import ProfilingCallback, profile_model, profile_tree

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
    assert row.est_total_bytes > 0


def test_profile_model_grad(sub_models):
    diffuse, _ = sub_models
    row = profile_model(diffuse, grad=True, n=3)
    assert row.grad_compile_s > 0
    assert row.grad_runtime_s > 0


def test_profile_model_broken_init_propagates(sub_models):
    diffuse, _ = sub_models

    class Broken:
        domain = diffuse.domain

        def init(self, key):
            raise RuntimeError('custom initializer failed')

        def __call__(self, x):
            return diffuse(x)

    with pytest.raises(RuntimeError, match='custom initializer'):
        profile_model(Broken(), n=3)


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

    table = str(report)
    assert 'diffuse' in table and 'TOTAL (fused)' in table

    out = tmp_path / 'profile.json'
    report.to_json(out)
    payload = json.loads(out.read_text())
    assert len(payload['rows']) == 2
    assert payload['root']['name'] == 'TOTAL (fused)'


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


def test_profile_model_jvp_and_meta(sub_models):
    diffuse, _ = sub_models
    row = profile_model(diffuse, jvp=True, n=3, meta={'n_vis': 7})
    assert row.jvp_compile_s > 0
    assert row.jvp_runtime_s > 0
    assert row.grad_runtime_s is None
    assert row.meta == {'n_vis': 7}
    # CPU backend: no memory statistics, no flops estimate.
    assert row.peak_bytes is None
    assert row.intensity is None or row.intensity > 0


def test_report_meta_columns_and_markdown(sub_models, tmp_path):
    diffuse, points = sub_models
    report = profile_tree({'diffuse': diffuse, 'points': points}, n=3,
                          jvp=True, verbose=False,
                          meta={'diffuse': {'n_vis': 7}})
    table = str(report)
    header = table.splitlines()[0]
    assert 'n_vis' in header
    assert 'jvp_runtime_s' in header
    # never measured on this backend, so the column is dropped
    assert 'peak_bytes' not in header
    assert 'grad_runtime_s' not in header

    md = report.to_markdown()
    assert md.startswith('| name | n_vis |')
    assert '| :-- | --: |' in md

    out = tmp_path / 'profile.json'
    report.to_json(out)
    payload = json.loads(out.read_text())
    assert payload['rows'][0]['meta'] == {'n_vis': 7}
    assert payload['rows'][1]['meta'] == {}
    assert 'intensity' in payload['rows'][0]
