//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_CONVERSION_PASSES_H
#define CANCER_CONVERSION_PASSES_H

namespace mlir {
namespace CANCER {

// Registers all CANCER conversion passes.
void registerConversionPasses();

} // namespace CANCER
} // namespace mlir

#endif // CANCER_CONVERSION_PASSES_H
