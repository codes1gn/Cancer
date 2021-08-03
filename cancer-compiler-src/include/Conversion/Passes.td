//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_CONVERSION_PASSES
#define CANCER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// Basicpy conversions
//===----------------------------------------------------------------------===//

def ConvertBasicpyToStd : Pass<"convert-basicpy-to-std", "FuncOp"> {
  let summary = "Convert representable Basicpy ops to std";
  let constructor = "mlir::CANCER::createConvertBasicpyToStdPass()";
}

//===----------------------------------------------------------------------===//
// Numpy conversions
//===----------------------------------------------------------------------===//

def ConvertNumpyToTCF : Pass<"convert-numpy-to-tcf", "FuncOp"> {
  let summary = "Convert the numpy dialect to supported TCF ops";
  let constructor = "mlir::CANCER::createConvertNumpyToTCFPass()";
}

//===----------------------------------------------------------------------===//
// TCFToTCP
//===----------------------------------------------------------------------===//

def ConvertTCFToLinalg : Pass<"convert-tcf-to-linalg", "FuncOp"> {
  let summary = "Convert TCF to Linalg";
  let description = [{
    The intention is for this pass to convert mainly to linalg named ops.

    Because linalg is at the "TCP" layer of abstraction, this pass has to
    concern itself with generating guards for error cases.
  }];
  let constructor = "mlir::CANCER::createConvertTCFToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// TCFToStd
//===----------------------------------------------------------------------===//

def ConvertTCFToStd : Pass<"convert-tcf-to-std", "FuncOp"> {
  let summary = "Convert TCF to Std";
  let constructor = "mlir::CANCER::createConvertTCFToStdPass()";
}

//===----------------------------------------------------------------------===//
// TCFToTCP
//===----------------------------------------------------------------------===//

def ConvertTCFToTCP : Pass<"convert-tcf-to-tcp", "FuncOp"> {
  let summary = "Convert TCF to TCP";
  let constructor = "mlir::CANCER::createConvertTCFToTCPPass()";
}

#endif // CANCER_CONVERSION_PASSES
