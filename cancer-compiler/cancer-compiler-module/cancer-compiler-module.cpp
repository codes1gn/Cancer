//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility binary for compiling and running code through the cancer
// compiler/runtime stack.
//
//===----------------------------------------------------------------------===//

#include "Capi/cancer-runner-backend.h"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cancer_compiler_module, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cancer_compiler_module

        .. autosummary::
           :toctree: _generate
          
           cancerrun
    )pbdoc";
  m.def(
      // TODO, change interface
      "cancerrun",
      [](std::vector<std::string> args) {
        std::vector<char *> cstrs;
        cstrs.reserve(args.size());
        for (auto &s : args) {
          cstrs.push_back(const_cast<char *>(s.c_str()));
        }
        return cancerrun(cstrs.size(), cstrs.data());
      },
      R"pbdoc(
          Run mlir
          python module
        )pbdoc");
}
