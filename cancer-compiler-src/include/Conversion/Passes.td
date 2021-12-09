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

def ConvertNumpyToAtir : Pass<"convert-numpy-to-atir", "FuncOp"> {
  let summary = "Convert the numpy dialect to supported Atir dialect";
  let constructor = "mlir::CANCER::createConvertNumpyToAtirPass()";
}

//===----------------------------------------------------------------------===//
// AtirToCtir
//===----------------------------------------------------------------------===//

def ConvertAtirToLinalg : Pass<"convert-atir-to-linalg", "FuncOp"> {
  let summary = "Convert Atir to Linalg";
  let description = [{
    The intention is for this pass to convert mainly to linalg named ops.

    Because linalg is at the "Ctir" layer of abstraction, this pass has to
    concern itself with generating guards for error cases.
  }];
  let constructor = "mlir::CANCER::createConvertAtirToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// AtirToStd
//===----------------------------------------------------------------------===//

def ConvertAtirToStd : Pass<"convert-atir-to-std", "FuncOp"> {
  let summary = "Convert Atir to Std";
  let constructor = "mlir::CANCER::createConvertAtirToStdPass()";
}

//===----------------------------------------------------------------------===//
// AtirToCtir
//===----------------------------------------------------------------------===//

def ConvertAtirToCtir : Pass<"convert-atir-to-ctir", "FuncOp"> {
  let summary = "Convert Atir to Ctir";
  let constructor = "mlir::CANCER::createConvertAtirToCtirPass()";
}

#endif // CANCER_CONVERSION_PASSES
