//===- CancerOps.cpp - Cancer dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Cancer/CancerOps.h"
#include "Dialect/Cancer/CancerDialect.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// ConstantOp

#define GET_OP_CLASSES
#include "Dialect/Cancer/CancerOps.cpp.inc"
