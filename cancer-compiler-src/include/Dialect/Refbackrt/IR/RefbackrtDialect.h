//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_DIALECT_REFBACKRT_IR_REFBACKRTDIALECT_H
#define CANCER_DIALECT_REFBACKRT_IR_REFBACKRTDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace CANCER {
namespace refbackrt {

class TensorType : public Type::TypeBase<TensorType, Type, TypeStorage> {
public:
  using Base::Base;

  static TensorType get(MLIRContext *context) { return Base::get(context); }
};

} // namespace refbackrt
} // namespace CANCER
} // namespace mlir

#include "Dialect/Refbackrt/IR/RefbackrtOpsDialect.h.inc"

#endif // CANCER_DIALECT_REFBACKRT_IR_REFBACKRTDIALECT_H
