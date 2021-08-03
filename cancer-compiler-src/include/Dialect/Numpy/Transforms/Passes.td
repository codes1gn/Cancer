//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_NUMPY_PASSES
#define CANCER_NUMPY_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def NumpyPublicFunctionsToTensor : Pass<"numpy-public-functions-to-tensor", "ModuleOp"> {
  let summary = "Converts public functions to operate on tensors (instead of ndarray)";
  let constructor = "mlir::CANCER::Numpy::createPublicFunctionsToTensorPass()";
}

#endif // CANCER_NUMPY_PASSES
