//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_CONVERSION_PASSDETAIL_H
#define CANCER_CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace CANCER {

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // namespace CANCER
} // end namespace mlir

#endif // CANCER_CONVERSION_PASSDETAIL_H
