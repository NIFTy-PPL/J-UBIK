// XLA FFI handler that executes an existing cufinufft plan.
//
// The plan is created, its points set and sorted, and eventually destroyed on
// the Python side (see _plan.py).  This handler only calls cufinufft_execute
// on it, so the plan (cuFFT plan, kernel spectrum, work arrays) and the bin
// sort of the points survive across likelihood evaluations instead of being
// rebuilt on every call.
//
// Everything an execution needs sits in one PlanHandle: the plan, the stream
// it was created on, the precision-specific execute function and the
// transform type.  The compiled program carries only the handle's address, so
// nothing in the HLO can disagree with the plan.
//
// No CUDA or cufinufft headers are needed: the CUDA runtime functions and the
// cufinufft_execute variant are handed in as function pointers, resolved in
// Python through ctypes from the libraries already loaded in the process.
// This keeps the build down to a plain C++ compiler and the XLA FFI headers
// that ship with jaxlib.
//
// Stream discipline: the plan is bound to its own CUDA stream at makeplan
// time (cufinufft cannot change it afterwards) while XLA hands us the stream
// the surrounding computation runs on.  Two events bracket the execute so
// that the plan stream waits for the inputs and XLA waits for the output.
// A per-plan mutex serializes this entire submission sequence: CUDA stream
// ordering alone does not protect the plan's mutable host state or prevent
// two executions from interleaving kernels that use the same workspace.

#include <Python.h>

#include <cstdint>
#include <mutex>
#include <new>
#include <string>

#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

namespace {

using ExecFn = int (*)(void* plan, void* c, void* f);
using EventCreateFn = int (*)(void** event, unsigned int flags);
using EventRecordFn = int (*)(void* event, void* stream);
using StreamWaitFn = int (*)(void* stream, void* event, unsigned int flags);
using EventDestroyFn = int (*)(void* event);

struct CudaEvents {
  EventCreateFn create = nullptr;
  EventRecordFn record = nullptr;
  StreamWaitFn stream_wait = nullptr;
  EventDestroyFn destroy = nullptr;
};

CudaEvents g_cuda;

// Owned by a Python capsule (see MakePlanHandle).  `plan` and `stream` are
// borrowed: the Python PlanSet destroys them after the handle is gone.
struct PlanHandle {
  void* plan;
  void* stream;
  ExecFn exec;
  int nufft_type;
  std::mutex mutex;
};

constexpr const char* kHandlerName = "cufinufft_exec";
constexpr const char* kPlanCapsule = "jubik.cufinufft.PlanHandle";
constexpr unsigned int kEventDisableTiming = 0x02;

ffi::Error CudaCheck(int code, const char* what) {
  if (code != 0) {
    return ffi::Error::Internal(std::string("cufinufft_exec: ") + what +
                                " failed with CUDA error " + std::to_string(code));
  }
  return ffi::Error::Success();
}

// Enqueue on `waiter` a wait for everything recorded so far on `recorder`.
ffi::Error Bridge(void* recorder, void* waiter) {
  void* event = nullptr;
  if (auto err = CudaCheck(g_cuda.create(&event, kEventDisableTiming), "cudaEventCreateWithFlags");
      err.failure()) {
    return err;
  }
  if (auto err = CudaCheck(g_cuda.record(event, recorder), "cudaEventRecord"); err.failure()) {
    g_cuda.destroy(event);
    return err;
  }
  if (auto err = CudaCheck(g_cuda.stream_wait(waiter, event, 0), "cudaStreamWaitEvent");
      err.failure()) {
    g_cuda.destroy(event);
    return err;
  }
  // Destroying an event that a stream still waits on is legal: the resource
  // is released once the wait has been consumed.
  return CudaCheck(g_cuda.destroy(event), "cudaEventDestroy");
}

ffi::Error Run(void* xla_stream, int64_t handle_ptr, ffi::AnyBuffer source,
               ffi::Result<ffi::AnyBuffer> out) {
  if (g_cuda.create == nullptr) {
    return ffi::Error::Internal("cufinufft_exec: not initialised, call _exec.init first");
  }
  auto* handle = reinterpret_cast<PlanHandle*>(handle_ptr);
  std::lock_guard<std::mutex> lock(handle->mutex);
  void* src = source.untyped_data();
  void* dst = out->untyped_data();

  if (auto err = Bridge(xla_stream, handle->stream); err.failure()) return err;

  // cufinufft_execute(plan, c, f): c lives on the points, f on the grid.
  int ret = (handle->nufft_type == 2) ? handle->exec(handle->plan, dst, src)
                                      : handle->exec(handle->plan, src, dst);
  if (ret != 0) {
    return ffi::Error::Internal("cufinufft_exec: cufinufft_execute failed with code " +
                                std::to_string(ret));
  }

  return Bridge(handle->stream, xla_stream);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufinufft_exec, Run,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .Attr<int64_t>("handle_ptr")
                                  .Arg<ffi::AnyBuffer>()    // source
                                  .Ret<ffi::AnyBuffer>());  // output

// -----------------------------------------------------------------------------
// Python module.  Plain C API, no binding library.
// -----------------------------------------------------------------------------

namespace {

void DeletePlanHandle(PyObject* capsule) {
  delete static_cast<PlanHandle*>(PyCapsule_GetPointer(capsule, kPlanCapsule));
}

// make_plan_handle(plan, stream, exec_fn, nufft_type) -> capsule
PyObject* MakePlanHandle(PyObject*, PyObject* args) {
  unsigned long long plan, stream, exec_fn;
  int nufft_type;
  if (!PyArg_ParseTuple(args, "KKKi", &plan, &stream, &exec_fn, &nufft_type)) return nullptr;
  if (nufft_type != 1 && nufft_type != 2) {
    PyErr_SetString(PyExc_ValueError, "nufft_type must be 1 or 2");
    return nullptr;
  }
  auto* handle = new (std::nothrow) PlanHandle{
      reinterpret_cast<void*>(plan), reinterpret_cast<void*>(stream),
      reinterpret_cast<ExecFn>(exec_fn), nufft_type, {}};
  if (handle == nullptr) return PyErr_NoMemory();
  PyObject* capsule = PyCapsule_New(handle, kPlanCapsule, DeletePlanHandle);
  if (capsule == nullptr) delete handle;
  return capsule;
}

PyObject* PlanHandleAddress(PyObject*, PyObject* capsule) {
  void* handle = PyCapsule_GetPointer(capsule, kPlanCapsule);
  if (handle == nullptr) return nullptr;
  return PyLong_FromVoidPtr(handle);
}

// init(event_create, event_record, stream_wait, event_destroy)
PyObject* Init(PyObject*, PyObject* args) {
  unsigned long long p[4];
  if (!PyArg_ParseTuple(args, "KKKK", &p[0], &p[1], &p[2], &p[3])) return nullptr;
  g_cuda.create = reinterpret_cast<EventCreateFn>(p[0]);
  g_cuda.record = reinterpret_cast<EventRecordFn>(p[1]);
  g_cuda.stream_wait = reinterpret_cast<StreamWaitFn>(p[2]);
  g_cuda.destroy = reinterpret_cast<EventDestroyFn>(p[3]);
  Py_RETURN_NONE;
}

PyObject* Handler(PyObject*, PyObject*) {
  return PyCapsule_New(reinterpret_cast<void*>(cufinufft_exec), nullptr, nullptr);
}

PyMethodDef kMethods[] = {
    {"make_plan_handle", MakePlanHandle, METH_VARARGS,
     "make_plan_handle(plan, stream, exec_fn, nufft_type): capsule owning the execution state."},
    {"plan_handle_address", PlanHandleAddress, METH_O,
     "Address of a plan handle for use as the FFI attribute."},
    {"init", Init, METH_VARARGS, "init(event_create, event_record, stream_wait, event_destroy)"},
    {"handler", Handler, METH_NOARGS, "The FFI handler as a PyCapsule, register under HANDLER_NAME."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {PyModuleDef_HEAD_INIT, "_exec", nullptr, -1, kMethods,
                       nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__exec(void) {
  PyObject* module = PyModule_Create(&kModule);
  if (module == nullptr) return nullptr;
  if (PyModule_AddStringConstant(module, "JAXLIB_VERSION", JUBIK_JAXLIB_VERSION) != 0 ||
      PyModule_AddStringConstant(module, "HANDLER_NAME", kHandlerName) != 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
