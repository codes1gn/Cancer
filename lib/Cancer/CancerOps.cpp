//===- CancerOps.cpp - Cancer dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cancer/CancerOps.h"
#include "Cancer/CancerDialect.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// ConstantOp

#define GET_OP_CLASSES
#include "Cancer/CancerOps.cpp.inc"
