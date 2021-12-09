//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_ATIR_PASSES
#define CANCER_ATIR_PASSES

include "mlir/Pass/PassBase.td"

def AtirShapeRefinement : Pass<"atir-shape-refinement", "FuncOp"> {
  let summary = "Refines shapes of tensors";
  let constructor = "mlir::CANCER::atir::createShapeRefinementPass()";
}

#endif // CANCER_ATIR_PASSES
