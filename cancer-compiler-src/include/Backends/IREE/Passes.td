//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_BACKEND_IREE_PASSES
#define CANCER_BACKEND_IREE_PASSES

include "mlir/Pass/PassBase.td"

def LowerLinkage : Pass<"cancer-iree-backend-lower-linkage", "ModuleOp"> {
  let summary = "Lower linkage of symbols to IREE's interfaces";
  let description = [{
    IREE has a specific input modeling for linkage, which differs from
    MLIR's native modeling. This pass lowers MLIR's native linkage constructs
    (which are otherwise used by cancer) into the form IREE requires.
  }];
  let constructor = "mlir::CANCER::IREEBackend::createLowerLinkagePass()";
}

#endif // CANCER_BACKEND_IREE_PASSES
