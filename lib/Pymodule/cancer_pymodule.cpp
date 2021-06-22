//===- NpcompModule.cpp - MLIR Python bindings ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unordered_map>

#include "capi/InitLLVM.h"

PYBIND11_MODULE(pycancer, m) {
  m.doc() = "Cancer native python bindings";

  m.def("register_all_dialects", ::cancerRegisterAllDialects);
  m.def("register_all_passes", ::cancerRegisterAllPasses);
  m.def("initialize_llvm_codegen", ::cancerInitializeLLVMCodegen);
}
