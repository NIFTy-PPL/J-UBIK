from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Protocol, Sequence, Literal, Iterable
from ..parse.data.data_loader import LoadingMode, IndexAndPath


class HasIndexBundle(Protocol):
    index: int  # the only field the dispatcher relies on


class LoadOne(Protocol):
    """Any callable that has the following

    Parameters
    ----------
    filepath: IndexAndPath
    *args: Any, number of further positional arguments
    **kwargs: Any, number of further keyword arguments

    Returns
    -------
    bundle: HasIndexBundle
        A bundle of data, with an index key.
    """

    def __call__(
        self, filepath: IndexAndPath, *args: Any, **kwargs: Any
    ) -> HasIndexBundle: ...


def run_concurrent_load(
    filepaths: Sequence[IndexAndPath],
    load_one: LoadOne,  # Return Boundle that has index
    *,
    extra_pos_args: tuple[Any, ...] = (),
    extra_kw_args: dict[str, Any] | None = None,
    workers: int | None = None,
    mode: LoadingMode = LoadingMode.PROCESSES,
) -> List[HasIndexBundle]:
    """
    Execute `load_one(index, filepath, *extra_pos_args, **extra_kw_args)`
    concurrently for every filepath.  The returned list is ordered so that
    `results[i]` corresponds to `filepaths[i]`.

    Parameters
    ----------
    filepaths        : iterable of path strings
    load_one         : a function following the `LoadOne` protocol, in particular it
                       must return a bundle that `HasIndexBundle`.
    extra_pos_args   : additional positional args passed to every call
    extra_kw_args    : additional keyword args passed to every call
    workers          : number of threads / processes (None = executor default)
    mode             : threads (I/O-bound); processes (CPU-bound)
    """
    extra_kw_args = extra_kw_args or {}
    Exec = ProcessPoolExecutor if mode == LoadingMode.PROCESSES else ThreadPoolExecutor
    print(Exec)

    # Pre-allocate slots; used to keep the order of filepaths
    results: list[HasIndexBundle | None] = [None] * len(filepaths)

    def _submit(pool, filepath: IndexAndPath):
        return pool.submit(load_one, filepath, *extra_pos_args, **extra_kw_args)

    with Exec(max_workers=workers) as pool:
        futures = [_submit(pool, fp) for fp in filepaths]

        for fut in as_completed(futures):
            boundle = fut.result()  # may raise if worker failed
            results[boundle.index] = boundle

    # mypy/pylint appeasement: strip Nones (all slots should be filled)
    return [r for r in results]


# ----------------------------------------------------------------------
# 4)  Thin wrapper that also supports “serial” mode
# ----------------------------------------------------------------------
def load_bundles(
    filepaths: tuple[IndexAndPath],
    load_one: LoadOne,
    *,
    extra_pos_args: tuple[Any, ...] = (),
    extra_kw_args: dict[str, Any] | None = None,
    mode: LoadingMode = LoadingMode.SERIAL,
    workers: int | None = None,
) -> Iterable[HasIndexBundle]:
    """
    • mode=='serial'   → generator (no threads/processes)
    • mode=='threads'  → ThreadPoolExecutor
    • mode=='processes'→ ProcessPoolExecutor
    """
    if mode == LoadingMode.SERIAL:
        return (
            load_one(fp, *extra_pos_args, **(extra_kw_args or {})) for fp in filepaths
        )

    return run_concurrent_load(
        filepaths,
        load_one,
        extra_pos_args=extra_pos_args,
        extra_kw_args=extra_kw_args,
        workers=workers,
        mode=mode,
    )
