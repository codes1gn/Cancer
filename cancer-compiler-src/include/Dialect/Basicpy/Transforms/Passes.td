//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_BASICPY_PASSES
#define CANCER_BASICPY_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def FunctionTypeInference : Pass<"basicpy-type-inference", "FuncOp"> {
  let summary = "Performs function level type inference";
  let constructor = "mlir::CANCER::Basicpy::createFunctionTypeInferencePass()";
}

#endif // CANCER_BASICPY_PASSES
