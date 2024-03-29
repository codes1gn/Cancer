//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Atir/IR/AtirDialect.h"
#include "Dialect/Atir/IR/AtirOps.h"

using namespace mlir;
using namespace mlir::CANCER::atir;

void AtirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Atir/IR/AtirOps.cpp.inc"
      >();
}
