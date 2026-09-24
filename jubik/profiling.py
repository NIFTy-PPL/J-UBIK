# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2026 Max-Planck-Society

# %%

import gc
import json
import subprocess
import time
from dataclasses import dataclass, asdict, field

import jax
import numpy as np
import nifty.re as jft


def _synthesize_input(model, key):
    """Draw a random input for `model` from its `.domain`.

    Uses `model.init(key)` when the model provides an initializer, so the
    benchmark input matches what the model expects; otherwise draws
    `jft.random_like` on the `.domain` pytree. Errors raised by an existing
    initializer propagate, a broken `init` must not silently change the
    profiled workload.
    """
    domain = getattr(model, 'domain', None)
    if domain is None:
        raise ValueError(
            "Model has no `.domain`; pass an explicit example input `x`. "
            "(Plain callables, e.g. the eROSITA response dict entries, "
            "carry no shape metadata.)")
    init = getattr(model, 'init', None)
    if callable(init):
        return init(key)
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


def _timed_compile(fun, *args):
    """AOT-compile `fun` for inputs `args`, returning (compiled, seconds)."""
    t0 = time.perf_counter()
    compiled = jax.jit(fun).lower(*args).compile()
    return compiled, time.perf_counter() - t0


def _device_stats(device):
    """`device.memory_stats()` or {} where the backend has none (CPU)."""
    try:
        return device.memory_stats() or {}
    except Exception:
        return {}


_NVIDIA_SMI_OK = None


def _device_used_bytes(device):
    """Device-wide used memory from `nvidia-smi`, or None.

    `device.memory_stats()` only sees XLA's own allocator. Workspace a
    custom call grabs with `cudaMalloc` (cufinufft plans and sort buffers,
    cuFFT plans) never enters those counters, but it does show up in the
    card's used memory. Device-wide, so on a shared card other processes
    leak in; with XLA preallocation on, the pool itself is constant and the
    growth between two readings is dominated by exactly those foreign
    allocations. It is a point reading: a workspace allocated and freed
    within one call is gone by the time we look. Costs one `nvidia-smi` call (tens of ms); disabled after
    the first failure.

    Only works when `nvidia-smi` sees a single GPU. A JAX device ordinal
    is not an `nvidia-smi` index: `CUDA_VISIBLE_DEVICES`,
    `JAX_CUDA_VISIBLE_DEVICES` and CUDA's default fastest-first order all
    remap it, so with more than one card the reading could silently
    describe another GPU. There it warns once and returns None. To support
    multi-GPU nodes, map the device to its PCI bus id (e.g.
    `cudaDeviceGetPCIBusId`) and query `nvidia-smi -i <bus id>`.
    """
    global _NVIDIA_SMI_OK
    if _NVIDIA_SMI_OK is False or getattr(device, 'platform', None) != 'gpu':
        return None
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=True).stdout
        lines = out.strip().splitlines()
        used = [int(float(line)) * 2 ** 20 for line in lines]
    except Exception:
        _NVIDIA_SMI_OK = False
        return None
    if len(used) != 1:
        _NVIDIA_SMI_OK = False
        jft.logger.warning(
            f'nvidia-smi reports {len(used)} GPUs; device-wide memory is '
            'only supported on single-GPU nodes, `device_growth_bytes` stays '
            'empty. See `jubik.profiling._device_used_bytes`.')
        return None
    _NVIDIA_SMI_OK = True
    return used[0]


def _best_of(compiled, args, n):
    """Min-of-n wall-clock runtime of `compiled(*args)`, blocking on every
    call."""
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def _est_total_bytes(mem):
    """XLA's total buffer footprint for the executable, or None.

    `argument + output + temp - alias` from `compiled.memory_analysis()`,
    the total JAX documents; aliased input/output buffers would otherwise
    count twice. This is a static breakdown of the executable's buffers,
    not a temporal runtime peak. An oversized intermediate shows up in it.
    """
    if mem is None:
        return None
    sizes = {a: getattr(mem, f'{a}_size_in_bytes', None)
             for a in ('argument', 'output', 'temp', 'alias')}
    if any(v is None for v in sizes.values()):
        return None
    return int(sizes['argument'] + sizes['output'] + sizes['temp']
               - sizes['alias'])


def _memory_analysis(compiled):
    """`compiled.memory_analysis()` or None where the backend lacks it."""
    try:
        return compiled.memory_analysis()
    except Exception:
        return None


def _peak_from_trace(trace, bytes_before, peak_before):
    """(`peak_bytes`, `peak_phase`) from the per-step allocator trace.

    The process peak only carries information about this row when the row
    raised it. Then `peak_bytes` is the final peak minus the row's baseline
    `bytes_in_use`, i.e. the row's high-water mark above what was already
    resident, and `peak_phase` is the last step whose checkpoint moved the
    peak. Otherwise both are None.
    """
    if bytes_before is None or peak_before is None or not trace:
        return None, None
    running = peak_before
    phase = None
    for step, counters in trace.items():
        peak = counters.get('peak_bytes_in_use')
        if peak is not None and peak > running:
            running, phase = peak, step
    if phase is None:
        return None, None
    return int(running - bytes_before), phase


def _device_growth(trace, used_before):
    """Max device-wide used memory over the trace minus the row's baseline."""
    if used_before is None:
        return None
    readings = [c.get('device_used_bytes') for c in trace.values()]
    readings = [r for r in readings if r is not None]
    if not readings:
        return None
    return int(max(readings) - used_before)


@dataclass
class ProfileRow:
    """Compile/runtime/memory numbers for one (sub-)model.

    Timings are measured. The memory and flop numbers are XLA compiler
    estimates for this model's own executable: `temp_bytes`,
    `argument_bytes` and `output_bytes` from `memory_analysis()`,
    `est_total_bytes` their total minus aliased buffers, `flops` and
    `bytes_accessed` from
    `cost_analysis()`. `flops`/`bytes_accessed` are typically None on
    the CPU backend.
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
    est_total_bytes: int = None
    jvp_compile_s: float = None
    jvp_runtime_s: float = None
    #: XLA's static estimates for the derivative executables, same
    #: meaning as the forward columns above.
    grad_flops: float = None
    grad_bytes_accessed: float = None
    grad_temp_bytes: int = None
    grad_est_total_bytes: int = None
    jvp_flops: float = None
    jvp_bytes_accessed: float = None
    jvp_temp_bytes: int = None
    jvp_est_total_bytes: int = None
    #: Allocator counters around the row (`device.memory_stats()`).
    #: `bytes_before`/`bytes_after`: `bytes_in_use` before the first compile
    #: and after this row's executables and tangent are dropped; their
    #: difference is what the row left resident. The input `x` is resident
    #: at both points. `peak_before`/`peak_after`: the process-wide
    #: `peak_bytes_in_use` at the same two points.
    bytes_before: int = None
    bytes_after: int = None
    peak_before: int = None
    peak_after: int = None
    #: `peak_after - bytes_before` when this row raised the process peak
    #: (then it is the row's exact high-water mark above its baseline);
    #: None when the peak is inherited from an earlier row or from model
    #: construction, because then nothing about this row is known.
    peak_bytes: int = None
    #: The step that last raised the process peak, one of
    #: `forward_compile`, `forward_run`, `grad_compile`, `grad_run`,
    #: `jvp_compile`, `jvp_run`; None when `peak_bytes` is None.
    peak_phase: str = None
    #: Device-wide used memory (`nvidia-smi`, outside XLA's counters) before
    #: the row, its maximum over the step checkpoints minus that baseline,
    #: and after the row. Read only at the checkpoints, between calls, so
    #: `device_growth_bytes` sees what a custom call allocates itself and
    #: keeps (cached cufinufft/cuFFT plans), not a workspace it frees again
    #: before returning. None without nvidia-smi or when it sees more than
    #: one GPU.
    device_used_before: int = None
    device_growth_bytes: int = None
    device_used_after: int = None
    #: `{step: {'bytes_in_use', 'peak_bytes_in_use', 'device_used_bytes'}}`
    #: after every step above, the raw trace the derived columns come from.
    peak_trace: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def intensity(self):
        """Arithmetic intensity, flops per byte accessed (XLA estimates).

        Low values mark memory-bound executables. None where either
        estimate is missing (CPU backend).
        """
        if not self.flops or not self.bytes_accessed:
            return None
        return self.flops / self.bytes_accessed


def profile_model(model, x=None, *, name=None, grad=False, jvp=False, n=50,
                  key=None, clear_caches=True, meta=None, device=None):
    """Profile one jax-compiled model: compile time, runtime, flops, memory.

    The model is jit-compiled in isolation via the AOT path
    (`jax.jit(model).lower(x).compile()`), so the numbers are clean
    per-model figures. They overcount relative to this model running
    fused inside a larger jit, because XLA fuses and eliminates work
    across sub-model boundaries.

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
    jvp : bool, optional
        Also compile and time the forward-mode derivative
        (`jax.jvp` along a random tangent). Linear-response transposes
        differ from their forward pass, so the two derivative modes can
        cost differently. Default False.
    n : int, optional
        Runtime is the minimum over `n` blocking calls. Default 50.
    key : jax PRNG key, optional
        Key for input synthesis. Default `PRNGKey(42)`.
    clear_caches : bool, optional
        Clear jax caches first so `compile_s` measures a real compile,
        not a cache hit. Default True.
    meta : dict, optional
        Free-form labels stored on the row (`ProfileRow.meta`), e.g. the
        number of visibilities a response maps to. Rendered as extra
        table columns and kept in the JSON.
    device : jax Device, optional
        Device the model runs on and whose `memory_stats()` feed
        `peak_bytes`. The input is synthesized on, or moved to, this
        device. Default `jax.devices()[0]`.

    Returns
    -------
    row : ProfileRow

    Notes
    -----
    Memory comes in two flavours. The static columns (`temp_bytes`,
    `est_total_bytes` and their `grad_`/`jvp_` twins) are XLA's buffer
    assignment for each executable: per row, deterministic, no allocator
    involved, but blind to workspace a custom call (cufinufft, cuDNN
    autotuning) allocates itself.

    The dynamic columns come from `device.memory_stats()`. Its
    `peak_bytes_in_use` is process-wide and never resets, so a row whose
    true peak stays below an earlier high-water mark (an earlier row, or
    model construction) learns nothing from it. `peak_bytes` is therefore
    `peak_after - bytes_before` only when `peak_after > peak_before`, and
    None otherwise; `peak_phase` names the step that set it and
    `peak_trace` keeps the counters after every step. Run rows in growing
    size order, in a fresh process, to get a value on most rows. All of
    them are None on backends without memory statistics (CPU).

    `device_growth_bytes` is checkpointed too: `nvidia-smi` after every
    step, not a peak during execution. It adds the foreign allocations
    still held at a checkpoint and misses transient ones.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if name is None:
        name = type(model).__name__
    if device is None:
        device = jax.devices()[0]
    # Execute where we monitor: `_timed_compile` follows the input's
    # placement, so synthesize on `device` and move a caller's `x` there.
    with jax.default_device(device):
        if x is None:
            x = _synthesize_input(model, key)
        x = jax.device_put(x, device)

    if clear_caches:
        jax.clear_caches()
    stats_before = _device_stats(device)
    bytes_before = stats_before.get('bytes_in_use')
    peak_before = stats_before.get('peak_bytes_in_use')
    device_used_before = _device_used_bytes(device)
    trace = {}

    def checkpoint(step):
        st = _device_stats(device)
        used = _device_used_bytes(device)
        if st or used is not None:
            trace[step] = {'bytes_in_use': st.get('bytes_in_use'),
                           'peak_bytes_in_use': st.get('peak_bytes_in_use'),
                           'device_used_bytes': used}

    compiled, compile_s = _timed_compile(model, x)
    cost = _cost_dict(compiled)
    mem = _memory_analysis(compiled)
    checkpoint('forward_compile')

    jax.block_until_ready(compiled(x))  # warmup, first call may still pay setup
    runtime_s = _best_of(compiled, (x,), n)
    checkpoint('forward_run')

    grad_compiled = jvp_compiled = None
    grad_compile_s = grad_runtime_s = None
    grad_cost, grad_mem = {}, None
    if grad:
        grad_fun = jax.grad(lambda p: _scalarize(model(p)))
        grad_compiled, grad_compile_s = _timed_compile(grad_fun, x)
        grad_cost = _cost_dict(grad_compiled)
        grad_mem = _memory_analysis(grad_compiled)
        checkpoint('grad_compile')
        jax.block_until_ready(grad_compiled(x))
        grad_runtime_s = _best_of(grad_compiled, (x,), n)
        checkpoint('grad_run')

    jvp_compile_s = jvp_runtime_s = None
    jvp_cost, jvp_mem = {}, None
    tangent = None
    if jvp:
        with jax.default_device(device):
            tangent = jft.random_like(jax.random.split(key)[1], x)
        # The tangent is a runtime argument. Closed over, it would be a
        # compile-time constant and XLA could fold a linear model's whole
        # derivative into it.
        jvp_fun = lambda p, t: jax.jvp(model, (p,), (t,))[1]
        jvp_compiled, jvp_compile_s = _timed_compile(jvp_fun, x, tangent)
        jvp_cost = _cost_dict(jvp_compiled)
        jvp_mem = _memory_analysis(jvp_compiled)
        checkpoint('jvp_compile')
        jax.block_until_ready(jvp_compiled(x, tangent))
        jvp_runtime_s = _best_of(jvp_compiled, (x, tangent), n)
        checkpoint('jvp_run')

    peak_bytes, peak_phase = _peak_from_trace(trace, bytes_before, peak_before)
    device_growth = _device_growth(trace, device_used_before)

    # Drop this row's executables and buffers before the next row compiles,
    # so rows do not pile up on a small card and `bytes_after` only shows
    # what outlives the call. `x` already sits in `bytes_before`.
    compiled = grad_compiled = jvp_compiled = None
    tangent = None
    gc.collect()
    stats_after = _device_stats(device)
    device_used_after = _device_used_bytes(device)

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
        est_total_bytes=_est_total_bytes(mem),
        jvp_compile_s=jvp_compile_s,
        jvp_runtime_s=jvp_runtime_s,
        grad_flops=grad_cost.get('flops'),
        grad_bytes_accessed=grad_cost.get('bytes accessed'),
        grad_temp_bytes=getattr(grad_mem, 'temp_size_in_bytes', None),
        grad_est_total_bytes=_est_total_bytes(grad_mem),
        jvp_flops=jvp_cost.get('flops'),
        jvp_bytes_accessed=jvp_cost.get('bytes accessed'),
        jvp_temp_bytes=getattr(jvp_mem, 'temp_size_in_bytes', None),
        jvp_est_total_bytes=_est_total_bytes(jvp_mem),
        bytes_before=bytes_before,
        bytes_after=stats_after.get('bytes_in_use'),
        peak_before=peak_before,
        peak_after=stats_after.get('peak_bytes_in_use'),
        peak_bytes=peak_bytes,
        peak_phase=peak_phase,
        device_used_before=device_used_before,
        device_growth_bytes=device_growth,
        device_used_after=device_used_after,
        peak_trace=trace,
        meta=dict(meta or {}),
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
    if b < 0:
        return '-' + _fmt_bytes(-b)
    if b < 2 ** 20:
        return f'{b / 2**10:.1f}KB'
    if b < 2 ** 30:
        return f'{b / 2**20:.1f}MB'
    return f'{b / 2**30:.2f}GB'


def _fmt_meta(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.3g}'
    return str(v)


def _fmt_intensity(i):
    return '-' if i is None else f'{i:.2f}'


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
    Columns whose value is None on every row are dropped from the table
    (jvp and peak memory where not measured), and every key found in any
    row's `meta` becomes a column right after the name.
    """

    _COLUMNS = (
        ('name', str, '<'),
        ('n_params', _fmt_count, '>'),
        ('compile_s', _fmt_seconds, '>'),
        ('runtime_s', _fmt_seconds, '>'),
        ('grad_runtime_s', _fmt_seconds, '>'),
        ('jvp_runtime_s', _fmt_seconds, '>'),
        ('flops', _fmt_count, '>'),
        ('intensity', _fmt_intensity, '>'),
        ('temp_bytes', _fmt_bytes, '>'),
        ('output_bytes', _fmt_bytes, '>'),
        ('est_total_bytes', _fmt_bytes, '>'),
        ('grad_est_total_bytes', _fmt_bytes, '>'),
        ('jvp_est_total_bytes', _fmt_bytes, '>'),
        ('peak_bytes', _fmt_bytes, '>'),
        ('peak_phase', _fmt_meta, '>'),
        ('device_growth_bytes', _fmt_bytes, '>'),
    )
    _ALWAYS = ('name',)

    def __init__(self, rows, root=None):
        self.rows = list(rows)
        self.root = root

    def _all_rows(self):
        return self.rows + ([self.root] if self.root else [])

    def _columns(self):
        """(header, getter, formatter, align) per rendered column."""
        all_rows = self._all_rows()
        meta_keys = []
        for r in all_rows:
            for k in r.meta:
                if k not in meta_keys:
                    meta_keys.append(k)
        cols = []
        for key, fmt, align in self._COLUMNS:
            if key not in self._ALWAYS and all(
                    getattr(r, key) is None for r in all_rows):
                continue
            cols.append((key, (lambda r, k=key: getattr(r, k)), fmt, align))
            if key == 'name':
                for mk in meta_keys:
                    cols.append((mk, (lambda r, k=mk: r.meta.get(k)),
                                 _fmt_meta, '>'))
        return cols

    def __str__(self):
        all_rows = self._all_rows()
        cols = self._columns()
        header = [c[0] for c in cols]
        table = [[fmt(get(r)) for _, get, fmt, _ in cols] for r in all_rows]
        widths = [max([len(h), *(len(t[i]) for t in table)])
                  for i, h in enumerate(header)]
        aligns = [c[3] for c in cols]
        lines = ['  '.join(f'{h:{a}{w}}' for h, a, w
                           in zip(header, aligns, widths))]
        lines.append('  '.join('-' * w for w in widths))
        for r, t in zip(all_rows, table):
            if r is self.root:
                lines.append('  '.join('-' * w for w in widths))
            lines.append('  '.join(f'{v:{a}{w}}' for v, a, w
                                   in zip(t, aligns, widths)))
        return '\n'.join(lines)

    def to_markdown(self):
        """The same table as a GitHub-flavoured markdown table."""
        all_rows = self._all_rows()
        cols = self._columns()
        header = [c[0] for c in cols]
        sep = [':--' if a == '<' else '--:' for *_, a in cols]
        lines = ['| ' + ' | '.join(header) + ' |',
                 '| ' + ' | '.join(sep) + ' |']
        for r in all_rows:
            cells = [fmt(get(r)) for _, get, fmt, _ in cols]
            if r is self.root:
                cells = [f'**{c}**' for c in cells]
            lines.append('| ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)

    @staticmethod
    def _row_dict(row):
        d = asdict(row)
        d['intensity'] = row.intensity
        return d

    def to_json(self, path):
        rows = [self._row_dict(r) for r in self.rows]
        root = self._row_dict(self.root) if self.root else None
        with open(path, 'w') as f:
            json.dump({'rows': rows, 'root': root}, f, indent=2)


def profile_tree(named_models, root=None, *, inputs=None, grad=False,
                 jvp=False, n=50, key=None, verbose=True, meta=None):
    """Profile a tree of named sub-models plus, optionally, the fused root.

    Each sub-model is jit-compiled and measured in isolation (see
    `profile_model` for the fusion caveat). The root, i.e. the full
    composed model as `optimize_kl` would jit it, is measured the same
    way and appended as the last row. Sub-models may nest, so do not add
    up rows; compare each row against the root instead.

    Parameters
    ----------
    named_models : dict[str, model]
        Sub-models to profile, keyed by display name.
    root : model, optional
        The full composed model.
    inputs : dict[str, pytree], optional
        Explicit example inputs per name (required for entries without
        `.domain`). Use key 'root' for the root model.
    grad, jvp, n, key
        Forwarded to `profile_model`.
    verbose : bool, optional
        Log each row as it is measured. Default True.
    meta : dict[str, dict], optional
        Per-name free-form labels, forwarded as `profile_model(meta=...)`.
        Use key 'root' for the root model.

    Returns
    -------
    report : ProfileReport
    """
    inputs = inputs or {}
    meta = meta or {}
    rows = []
    for name, model in named_models.items():
        row = profile_model(model, inputs.get(name), name=name, grad=grad,
                            jvp=jvp, n=n, key=key, meta=meta.get(name))
        if verbose:
            jft.logger.info(
                f'profiled {name}: compile {_fmt_seconds(row.compile_s)}, '
                f'run {_fmt_seconds(row.runtime_s)}')
        rows.append(row)
    root_row = None
    if root is not None:
        root_row = profile_model(root, inputs.get('root'), name='TOTAL (fused)',
                                 grad=grad, jvp=jvp, n=n, key=key,
                                 meta=meta.get('root'))
    return ProfileReport(rows, root_row)


class ProfilingCallback:
    """`optimize_kl` callback: per-iteration wall time and device memory.

    Append to the callback chain (signature `callback(samples, state)`).
    Writes one JSON line per iteration to `path` (if given) and logs a
    one-line summary, so memory growth and iteration-time jumps (e.g.
    from recompiles on sample-mode switches) are visible over the run.

    Wall time is measured between successive invocations, so the first
    call records only a baseline. The memory fields come from
    `device.memory_stats()`, which the CPU backend does not provide; there
    they are None and print as `-`.
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
