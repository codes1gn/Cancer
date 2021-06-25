#include <cstddef>
#include <pybind11/pybind11.h>
#include <unordered_map>

#include "Capi/InitLLVM.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(cancer_compiler, m) {
  m.doc() = R"pbdoc(
        Cancer python module bindings
        -----------------------------

        .. currentmodule:: cancer_compiler

        .. autosummary::
           :toctree: _generate

           __version__
           register_all_dialects
           register_all_passes
           initialize_llvm_codegen
    )pbdoc";

  // m.def("register_all_dialects", ::cancerRegisterAllDialects);
  // m.def("register_all_passes", ::cancerRegisterAllPasses);
  m.def("initialize_llvm_codegen", ::cancerInitializeLLVMCodegen,
        R"pbdoc(
        api that inits the llvm_codegen backend, which calls into LLVMInit.h
        )pbdoc");
  m.def("add", &add);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
