//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LaLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_CONVERSION_NUMPYTOTCF_PASSES_H
#define CANCER_CONVERSION_NUMPYTOTCF_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace CANCER {
std::unique_ptr<OperationPass<FuncOp>> createConvertNumpyToTCFPass();
}
} // namespace mlir

#endif // CANCER_CONVERSION_NUMPYTOTCF_PASSES_H
