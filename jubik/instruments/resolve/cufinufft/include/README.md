# Vendored XLA FFI headers

`xla/ffi/api/{api.h,c_api.h,ffi.h}` are copied unchanged from the jaxlib
0.6.2 wheel (`jaxlib/include/xla/ffi/api`, XLA FFI API version 0.1). They
are the only headers `_exec.cpp` needs, and vendoring them lets jubik build
the handler without jax in the build environment. They are Apache-2.0
licensed by the OpenXLA Authors, see the header of each file.

XLA loads handlers built against older FFI headers than its own and rejects
newer ones, so these headers set the oldest jaxlib the handler runs on. The
`resolve-cuda` extra in `pyproject.toml` pins `jaxlib` to at least this
version. `c_api.h` promises support for old API versions for at least 12
months; when a new jaxlib stops loading the handler, replace these files
with the headers of a newer jaxlib and raise the pin to match.
