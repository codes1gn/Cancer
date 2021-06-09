//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CancerInit.h"

#include "Dialect/Basicpy/IR/BasicpyDialect.h"
#include "Dialect/Basicpy/Transforms/Passes.h"
#include "Dialect/Numpy/IR/NumpyDialect.h"
#include "Dialect/Numpy/Transforms/Passes.h"
#include "Dialect/Refback/IR/RefbackDialect.h"
#include "Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "Dialect/TCF/IR/TCFDialect.h"
#include "Dialect/TCF/Transforms/Passes.h"
#include "Dialect/TCP/IR/TCPDialect.h"
#include "Dialect/TCP/Transforms/Passes.h"

#include "Conversion/Passes.h"

void mlir::CANCER::registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<Basicpy::BasicpyDialect>();
  registry.insert<Numpy::NumpyDialect>();
  registry.insert<tcf::TCFDialect>();
  registry.insert<tcp::TCPDialect>();
  registry.insert<refback::RefbackDialect>();
  registry.insert<refbackrt::RefbackrtDialect>();
  // clang-format on
}

void mlir::CANCER::registerAllPasses() {
  mlir::CANCER::registerBasicpyPasses();
  mlir::CANCER::registerNumpyPasses();
  mlir::CANCER::registerTCFPasses();
  mlir::CANCER::registerTCPPasses();
  mlir::CANCER::registerConversionPasses();
}
