# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2026 Max-Planck-Society

# %%

import json
import time
from dataclasses import dataclass, asdict

import jax
import numpy as np
import nifty.re as jft


def _synthesize_input(model, key):
    """Draw a random input for `model` from its `.domain`.

    Prefers `model.init(key)` (respects the model's own initializer);
    falls back to `jft.random_like` on the `.domain` pytree.
    """
    domain = getattr(model, 'domain', None)
    if domain is None:
        raise ValueError(
            "Model has no `.domain`; pass an explicit example input `x`. "
            "(Plain callables, e.g. the eROSITA response dict entries, "
            "carry no shape metadata.)")
    try:
        return model.init(key)
    except Exception:
        return jft.random_like(key, domain)


def _scalarize(out):
    """Reduce an arbitrary (possibly complex) pytree output to a real scalar."""
    leaves = jax.tree_util.tree_leaves(out)
    return sum(jax.numpy.sum(jax.numpy.abs(leaf) ** 2) for leaf in leaves)


def _cost_dict(compiled):
    """`compiled.cost_analysis()` normalized to a dict.

    Returns {} where the backend does not populate it (notably CPU).
    Older jax versions return a list with a single dict.
    """
    try:
        cost = compiled.cost_analysis()
    except Exception:
        return {}
    if isinstance(cost, (list, tuple)):
        cost = cost[0] if cost else None
    return dict(cost) if cost else {}


def _timed_compile(fun, x):
    """AOT-compile `fun` for input `x`, returning (compiled, seconds)."""
    t0 = time.perf_counter()
    compiled = jax.jit(fun).lower(x).compile()
    return compiled, time.perf_counter() - t0


def _best_of(compiled, x, n):
    """Min-of-n wall-clock runtime, blocking on every call."""
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(x))
        best = min(best, time.perf_counter() - t0)
    return best


def _peak_bytes(device):
    """Measured peak device memory, or None where unsupported (CPU)."""
    try:
        stats = device.memory_stats()
        return stats.get('peak_bytes_in_use') if stats else None
    except Exception:
        return None


@dataclass
class ProfileRow:
    """Compile/runtime/memory numbers for one (sub-)model.

    Static numbers (`flops`, `bytes_accessed`, `*_bytes` from
    `memory_analysis`) are XLA compiler estimates on the fused executable;
    `flops`/`bytes_accessed` are typically None on the CPU backend.
    `peak_bytes` is the device allocator's high-water mark and is only
    meaningful relative to other rows measured in the same process.
    """
    name: str
    n_params: int = None
    compile_s: float = None
    runtime_s: float = None
    grad_compile_s: float = None
    grad_runtime_s: float = None
    flops: float = None
    bytes_accessed: float = None
    temp_bytes: int = None
    argument_bytes: int = None
    output_bytes: int = None
    peak_bytes: int = None


def profile_model(model, x=None, *, name=None, grad=False, n=50,
                  key=None, device=None, clear_caches=True):
    """Profile one jax-compiled model: compile time, runtime, flops, memory.

    The model is jit-compiled in isolation via the AOT path
    (`jax.jit(model).lower(x).compile()`), so the numbers are clean
    per-model figures. Note they will overcount relative to this model
    running fused inside a larger jit — XLA fuses and eliminates work
    across sub-model boundaries. Compare with a `profile_tree` root row
    to see the fusion gap.

    Parameters
    ----------
    model : jft.Model, jft.Likelihood or callable
        Model to profile. Anything with a `.domain` can synthesize its
        own input; a plain callable requires `x`.
    x : pytree, optional
        Example input. If None, drawn from `model.domain`.
    name : str, optional
        Row label. Defaults to the class name of `model`.
    grad : bool, optional
        Also compile and time `jax.grad` of the sum-of-squares of the
        output — inference is gradient-dominated, so the VJP cost often
        matters more than the forward. Default False.
    n : int, optional
        Runtime is the minimum over `n` blocking calls. Default 50.
    key : jax PRNG key, optional
        Key for input synthesis. Default `PRNGKey(42)`.
    device : jax.Device, optional
        Device whose `memory_stats` provide `peak_bytes`.
        Default `jax.devices()[0]`.
    clear_caches : bool, optional
        Clear jax caches first so `compile_s` measures a real compile,
        not a cache hit. Default True.

    Returns
    -------
    row : ProfileRow
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if device is None:
        device = jax.devices()[0]
    if x is None:
        x = _synthesize_input(model, key)
    if name is None:
        name = type(model).__name__

    if clear_caches:
        jax.clear_caches()

    compiled, compile_s = _timed_compile(model, x)
    cost = _cost_dict(compiled)
    try:
        mem = compiled.memory_analysis()
    except Exception:
        mem = None

    jax.block_until_ready(compiled(x))  # warmup, first call may still pay setup
    runtime_s = _best_of(compiled, x, n)

    grad_compile_s = grad_runtime_s = None
    if grad:
        grad_fun = jax.grad(lambda p: _scalarize(model(p)))
        grad_compiled, grad_compile_s = _timed_compile(grad_fun, x)
        jax.block_until_ready(grad_compiled(x))
        grad_runtime_s = _best_of(grad_compiled, x, n)

    domain = getattr(model, 'domain', None)
    return ProfileRow(
        name=name,
        n_params=int(jft.size(domain)) if domain is not None else None,
        compile_s=compile_s,
        runtime_s=runtime_s,
        grad_compile_s=grad_compile_s,
        grad_runtime_s=grad_runtime_s,
        flops=cost.get('flops'),
        bytes_accessed=cost.get('bytes accessed'),
        temp_bytes=getattr(mem, 'temp_size_in_bytes', None),
        argument_bytes=getattr(mem, 'argument_size_in_bytes', None),
        output_bytes=getattr(mem, 'output_size_in_bytes', None),
        peak_bytes=_peak_bytes(device),
    )


def _fmt_seconds(s):
    if s is None:
        return '-'
    if s < 1e-3:
        return f'{s * 1e6:.0f}us'
    if s < 1.:
        return f'{s * 1e3:.2f}ms'
    return f'{s:.2f}s'


def _fmt_bytes(b):
    if b is None:
        return '-'
    if b < 2 ** 20:
        return f'{b / 2**10:.1f}KB'
    if b < 2 ** 30:
        return f'{b / 2**20:.1f}MB'
    return f'{b / 2**30:.2f}GB'


def _fmt_count(c):
    if c is None:
        return '-'
    if c < 1e6:
        return f'{c:.0f}'
    if c < 1e9:
        return f'{c / 1e6:.1f}M'
    return f'{c / 1e9:.2f}G'


class ProfileReport:
    """Result of `profile_tree`: per-sub-model rows plus optional root row.

    `str(report)` renders a table; `report.to_json(path)` persists it.
    """

    _COLUMNS = (
        ('name', str, '<'),
        ('n_params', _fmt_count, '>'),
        ('compile_s', _fmt_seconds, '>'),
        ('runtime_s', _fmt_seconds, '>'),
        ('grad_runtime_s', _fmt_seconds, '>'),
        ('flops', _fmt_count, '>'),
        ('temp_bytes', _fmt_bytes, '>'),
        ('output_bytes', _fmt_bytes, '>'),
        ('peak_bytes', _fmt_bytes, '>'),
    )

    def __init__(self, rows, root=None):
        self.rows = list(rows)
        self.root = root

    def __str__(self):
        all_rows = self.rows + ([self.root] if self.root else [])
        header = [c[0] for c in self._COLUMNS]
        table = [[fmt(getattr(r, key)) for key, fmt, _ in self._COLUMNS]
                 for r in all_rows]
        widths = [max(len(h), *(len(t[i]) for t in table))
                  for i, h in enumerate(header)]
        lines = ['  '.join(f'{h:{a}{w}}' for h, (_, _, a), w
                           in zip(header, self._COLUMNS, widths))]
        lines.append('  '.join('-' * w for w in widths))
        for r, t in zip(all_rows, table):
            if r is self.root:
                lines.append('  '.join('-' * w for w in widths))
            lines.append('  '.join(f'{v:{a}{w}}' for v, (_, _, a), w
                                   in zip(t, self._COLUMNS, widths)))
        gap = self.fusion_gap()
        if gap is not None:
            lines.append(
                f'sum(parts)/root runtime: {gap:.2f}x '
                '(>1 means XLA fused work across sub-model boundaries)')
        return '\n'.join(lines)

    def fusion_gap(self):
        """sum-of-parts runtime over root runtime; None without a root row."""
        if self.root is None or not self.root.runtime_s:
            return None
        parts = sum(r.runtime_s for r in self.rows if r.runtime_s)
        return parts / self.root.runtime_s

    def to_json(self, path):
        rows = [asdict(r) for r in self.rows]
        root = asdict(self.root) if self.root else None
        with open(path, 'w') as f:
            json.dump({'rows': rows, 'root': root}, f, indent=2)


def profile_tree(named_models, root=None, *, inputs=None, grad=False,
                 n=50, key=None, device=None, verbose=True):
    """Profile a tree of named sub-models plus, optionally, the fused root.

    Each sub-model is jit-compiled and measured in isolation
    (see `profile_model` for the fusion caveat); the root — the full
    composed model, as `optimize_kl` would jit it — is measured the same
    way and appended as the last row, so the report shows sum-of-parts
    against the fused whole.

    Parameters
    ----------
    named_models : dict[str, model]
        Sub-models to profile, keyed by display name.
    root : model, optional
        The full composed model.
    inputs : dict[str, pytree], optional
        Explicit example inputs per name (required for entries without
        `.domain`). Use key 'root' for the root model.
    grad, n, key, device
        Forwarded to `profile_model`.
    verbose : bool, optional
        Log each row as it is measured. Default True.

    Returns
    -------
    report : ProfileReport
    """
    inputs = inputs or {}
    rows = []
    for name, model in named_models.items():
        row = profile_model(model, inputs.get(name), name=name, grad=grad,
                            n=n, key=key, device=device)
        if verbose:
            jft.logger.info(
                f'profiled {name}: compile {_fmt_seconds(row.compile_s)}, '
                f'run {_fmt_seconds(row.runtime_s)}')
        rows.append(row)
    root_row = None
    if root is not None:
        root_row = profile_model(root, inputs.get('root'), name='TOTAL (fused)',
                                 grad=grad, n=n, key=key, device=device)
    return ProfileReport(rows, root_row)


def named_models_from_lens_system(system, skip_errors=True):
    """Enumerate a charm_lensing `CompiledSystem` into {address: jft.Model}.

    Duck-typed against the algebra access API (`system.paths()`,
    `system[address].model`), so jubik needs no charm_lensing import.
    Feed the result to `profile_tree`.

    Parameters
    ----------
    system : charm_lensing CompiledSystem
    skip_errors : bool, optional
        Skip addresses whose model cannot be built (e.g. empty slots)
        instead of raising. Default True.
    """
    named = {}
    for path in system.paths():
        try:
            model = system[path].model
        except Exception:
            if skip_errors:
                continue
            raise
        if model is not None:
            named[path] = model
    return named


class ProfilingCallback:
    """`optimize_kl` callback: per-iteration wall time and device memory.

    Append to the callback chain (signature `callback(samples, state)`).
    Writes one JSON line per iteration to `path` (if given) and logs a
    one-line summary, so memory growth and iteration-time jumps (e.g.
    from recompiles on sample-mode switches) are visible over the run.

    Wall time is measured between successive invocations, so the first
    call records only a baseline.
    """

    def __init__(self, path=None, device=None):
        self._path = path
        self._device = device if device is not None else jax.devices()[0]
        self._last = None

    def __call__(self, samples, state):
        now = time.perf_counter()
        wall_s = now - self._last if self._last is not None else None
        self._last = now
        try:
            stats = self._device.memory_stats() or {}
        except Exception:
            stats = {}
        record = {
            'nit': int(state.nit),
            'wall_s': wall_s,
            'bytes_in_use': stats.get('bytes_in_use'),
            'peak_bytes_in_use': stats.get('peak_bytes_in_use'),
            'n_live_arrays': len(jax.live_arrays()),
        }
        jft.logger.info(
            f'nit {record["nit"]}: wall {_fmt_seconds(wall_s)}, '
            f'in_use {_fmt_bytes(record["bytes_in_use"])}, '
            f'peak {_fmt_bytes(record["peak_bytes_in_use"])}, '
            f'live arrays {record["n_live_arrays"]}')
        if self._path is not None:
            with open(self._path, 'a') as f:
                f.write(json.dumps(record) + '\n')
