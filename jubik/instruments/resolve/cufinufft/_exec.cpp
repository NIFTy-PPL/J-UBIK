// XLA FFI handler that executes an existing cufinufft plan.
//
// The plan is created, its points set and sorted, and eventually destroyed on
// the Python side (see _handles.py).  This handler only calls
// cufinufft_execute on it, so the plan (cuFFT plan, kernel spectrum, work
// arrays) and the bin sort of the points survive across likelihood
// evaluations instead of being rebuilt on every call.
//
// No CUDA or cufinufft headers are needed: the four CUDA runtime functions and
// the two cufinufft_execute variants are handed in as function pointers by
// cufinufft_exec_init, resolved in Python through ctypes from the libraries
// that are already loaded in the process.  This keeps the build down to a
// plain C++ compiler and the XLA FFI headers that ship with jaxlib.
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

struct Runtime {
  ExecFn exec_f64 = nullptr;
  ExecFn exec_f32 = nullptr;
  EventCreateFn event_create = nullptr;
  EventRecordFn event_record = nullptr;
  StreamWaitFn stream_wait = nullptr;
  EventDestroyFn event_destroy = nullptr;
};

Runtime g_runtime;

// Owned by a Python capsule alongside the cufinufft Plan. Lowering retains
// their PlanSet in the executable, so this address remains valid at runtime.
struct PlanHandle {
  void* plan;
  void* stream;
  std::mutex mutex;
};

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
  if (auto err = CudaCheck(g_runtime.event_create(&event, kEventDisableTiming),
                           "cudaEventCreateWithFlags");
      err.failure()) {
    return err;
  }
  if (auto err = CudaCheck(g_runtime.event_record(event, recorder), "cudaEventRecord");
      err.failure()) {
    g_runtime.event_destroy(event);
    return err;
  }
  if (auto err = CudaCheck(g_runtime.stream_wait(waiter, event, 0), "cudaStreamWaitEvent");
      err.failure()) {
    g_runtime.event_destroy(event);
    return err;
  }
  // Destroying an event that a stream still waits on is legal: the resource
  // is released once the wait has been consumed.
  return CudaCheck(g_runtime.event_destroy(event), "cudaEventDestroy");
}

ffi::Error Run(void* xla_stream, int64_t handle_ptr, int64_t nufft_type,
               int64_t is_double, ffi::AnyBuffer source, ffi::Result<ffi::AnyBuffer> out) {
  if (g_runtime.exec_f64 == nullptr) {
    return ffi::Error::Internal(
        "cufinufft_exec: runtime not initialised, call cufinufft_exec_init first");
  }
  if (nufft_type != 1 && nufft_type != 2) {
    return ffi::Error::InvalidArgument("cufinufft_exec: nufft_type must be 1 or 2");
  }

  ExecFn exec = is_double ? g_runtime.exec_f64 : g_runtime.exec_f32;
  auto* handle = reinterpret_cast<PlanHandle*>(handle_ptr);
  std::lock_guard<std::mutex> lock(handle->mutex);
  void* plan = handle->plan;
  void* pstream = handle->stream;
  void* src = source.untyped_data();
  void* dst = out->untyped_data();

  if (auto err = Bridge(xla_stream, pstream); err.failure()) return err;

  // cufinufft_execute(plan, c, f): c lives on the points, f on the grid.
  int ret = (nufft_type == 2) ? exec(plan, dst, src) : exec(plan, src, dst);
  if (ret != 0) {
    return ffi::Error::Internal("cufinufft_exec: cufinufft_execute failed with code " +
                                std::to_string(ret));
  }

  return Bridge(pstream, xla_stream);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufinufft_exec, Run,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .Attr<int64_t>("handle_ptr")
                                  .Attr<int64_t>("nufft_type")
                                  .Attr<int64_t>("is_double")
                                  .Arg<ffi::AnyBuffer>()    // source
                                  .Ret<ffi::AnyBuffer>());  // output

// -----------------------------------------------------------------------------
// Python module: hands the handler out as a PyCapsule and takes the function
// addresses the handler needs.  Plain C API, no binding library.
// -----------------------------------------------------------------------------

namespace {

void DeletePlanHandle(PyObject* capsule) {
  delete static_cast<PlanHandle*>(PyCapsule_GetPointer(capsule, kPlanCapsule));
}

PyObject* MakePlanHandle(PyObject*, PyObject* args) {
  unsigned long long plan, stream;
  if (!PyArg_ParseTuple(args, "KK", &plan, &stream)) return nullptr;
  auto* handle = new (std::nothrow) PlanHandle{
      reinterpret_cast<void*>(plan), reinterpret_cast<void*>(stream), {}};
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

PyObject* Init(PyObject*, PyObject* args) {
  unsigned long long p[6];
  if (!PyArg_ParseTuple(args, "KKKKKK", &p[0], &p[1], &p[2], &p[3], &p[4], &p[5])) {
    return nullptr;
  }
  g_runtime.exec_f64 = reinterpret_cast<ExecFn>(p[0]);
  g_runtime.exec_f32 = reinterpret_cast<ExecFn>(p[1]);
  g_runtime.event_create = reinterpret_cast<EventCreateFn>(p[2]);
  g_runtime.event_record = reinterpret_cast<EventRecordFn>(p[3]);
  g_runtime.stream_wait = reinterpret_cast<StreamWaitFn>(p[4]);
  g_runtime.event_destroy = reinterpret_cast<EventDestroyFn>(p[5]);
  Py_RETURN_NONE;
}

PyObject* Registrations(PyObject*, PyObject*) {
  PyObject* dict = PyDict_New();
  if (dict == nullptr) return nullptr;
  PyObject* capsule = PyCapsule_New(reinterpret_cast<void*>(cufinufft_exec), nullptr, nullptr);
  if (capsule == nullptr || PyDict_SetItemString(dict, "cufinufft_exec", capsule) != 0) {
    Py_XDECREF(capsule);
    Py_DECREF(dict);
    return nullptr;
  }
  Py_DECREF(capsule);
  return dict;
}

PyMethodDef kMethods[] = {
    {"make_plan_handle", MakePlanHandle, METH_VARARGS,
     "Keep a plan address, stream and execution mutex in a capsule."},
    {"plan_handle_address", PlanHandleAddress, METH_O,
     "Address of a plan handle for use in an FFI attribute."},
    {"init", Init, METH_VARARGS,
     "init(exec_f64, exec_f32, event_create, event_record, stream_wait, event_destroy)"},
    {"registrations", Registrations, METH_NOARGS, "FFI handlers by name, as PyCapsules"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {PyModuleDef_HEAD_INIT, "_exec", nullptr, -1, kMethods,
                       nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__exec(void) {
  PyObject* module = PyModule_Create(&kModule);
  if (module == nullptr) return nullptr;
  if (PyModule_AddStringConstant(module, "JAXLIB_VERSION", JUBIK_JAXLIB_VERSION) != 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
