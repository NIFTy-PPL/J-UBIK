"""Persistent cufinufft plans as a JAX-callable radio response kernel.

Points are uploaded at construction; plans and point sorts are built lazily
during lowering, once per transform type, sign and batch size. Executables
retain their plans and reuse them across likelihood evaluations; only
``cufinufft_execute`` and stream coordination run inside the compiled graph.
Requires the ``cufinufft`` package and the compiled ``_exec`` handler.
"""

from ._handles import PlanSet
from ._primitive import nufft1, nufft2

__all__ = ["PlanSet", "nufft1", "nufft2"]
