import json

import jax
import jax.numpy as jnp
import nifty.re as jft
import pytest

from jubik.profiling import (ProfileReport, ProfilingCallback, profile_model,
                             profile_tree)

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


def test_peak_from_trace_only_when_the_row_raised_the_peak():
    from jubik.profiling import _peak_from_trace

    # inherited: no checkpoint moves above the pre-row process peak
    trace = {'forward_compile': {'bytes_in_use': 110, 'peak_bytes_in_use': 1000},
             'forward_run': {'bytes_in_use': 110, 'peak_bytes_in_use': 1000}}
    assert _peak_from_trace(trace, bytes_before=100, peak_before=1000) == (None, None)

    # raised during the grad run: exact increment above the row's baseline
    trace['grad_compile'] = {'bytes_in_use': 120, 'peak_bytes_in_use': 1000}
    trace['grad_run'] = {'bytes_in_use': 120, 'peak_bytes_in_use': 1500}
    trace['jvp_run'] = {'bytes_in_use': 120, 'peak_bytes_in_use': 1500}
    assert _peak_from_trace(trace, 100, 1000) == (1400, 'grad_run')

    # no statistics at all (CPU)
    assert _peak_from_trace({}, None, None) == (None, None)


def test_profile_model_derivative_static_memory(sub_models):
    diffuse, _ = sub_models
    row = profile_model(diffuse, grad=True, jvp=True, n=3)
    # static XLA estimates exist per executable on every backend
    assert row.grad_est_total_bytes > 0 and row.jvp_est_total_bytes > 0
    assert row.grad_temp_bytes is not None and row.jvp_temp_bytes is not None
    # CPU: no allocator statistics, so the dynamic columns stay None
    assert (row.bytes_before, row.peak_after, row.peak_bytes, row.peak_phase) == (None,) * 4
    assert row.peak_trace == {}
    d = ProfileReport._row_dict(row) if hasattr(ProfileReport, '_row_dict') else None
    assert d is None or 'peak_trace' in d


def test_empty_report_renders():
    report = ProfileReport([], None)
    assert str(report).splitlines()[0].strip() == 'name'
    assert report.to_markdown().startswith('| name |')


def test_device_growth_and_negative_bytes_format():
    from jubik.profiling import _device_growth, _fmt_bytes

    trace = {'a': {'device_used_bytes': 100}, 'b': {'device_used_bytes': 300},
             'c': {'device_used_bytes': None}}
    assert _device_growth(trace, 50) == 250
    assert _device_growth(trace, None) is None
    assert _device_growth({}, 50) is None
    assert _fmt_bytes(-3 * 2**20) == '-3.0MB'


def test_device_used_bytes_is_none_off_gpu():
    from jubik.profiling import _device_used_bytes

    assert _device_used_bytes(jax.devices('cpu')[0]) is None
