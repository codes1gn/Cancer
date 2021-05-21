//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_TYPING_TRANSFORMS_PASSES
#define CANCER_TYPING_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def CPAFunctionTypeInference : Pass<"cancer-cpa-type-inference", "FuncOp"> {
  let summary = "Performs CPA function level type inference";
  let constructor = "mlir::CANCER::Typing::createCPAFunctionTypeInferencePass()";
}

#endif // CANCER_TYPING_TRANSFORMS_PASSES
