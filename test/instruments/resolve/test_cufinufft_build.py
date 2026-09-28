"""The cufinufft FFI handler is compiled by every install.

Needs no GPU, so it runs where the GPU regressions in test_cufinufft.py skip:
a plain ``pip install .`` with a C++ compiler must leave ``_exec`` importable.
"""


def test_exec_handler_is_built():
    from jubik.instruments.resolve.cufinufft import _exec

    assert isinstance(_exec.HANDLER_NAME, str)
    assert type(_exec.handler()).__name__ == "PyCapsule"
