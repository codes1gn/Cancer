//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_CTIR_PASSES
#define CANCER_CTIR_PASSES

include "mlir/Pass/PassBase.td"

def CtirBufferize : Pass<"ctir-bufferize", "FuncOp"> {
  let summary = "Bufferizes the ctir dialect";
  let constructor = "mlir::CANCER::createCtirBufferizePass()";
}

#endif // CANCER_CTIR_PASSES
