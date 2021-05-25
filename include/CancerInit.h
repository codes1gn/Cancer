//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_INITALL_H
#define CANCER_INITALL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace CANCER {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace CANCER
} // namespace mlir

#endif // CANCER_INITALL_H
