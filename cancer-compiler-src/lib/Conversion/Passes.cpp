//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/Passes.h"

#include "Conversion/BasicpyToStd/Passes.h"
#include "Conversion/NumpyToTCF/Passes.h"
#include "Conversion/TCFToLinalg/TCFToLinalg.h"
#include "Conversion/TCFToStd/TCFToStd.h"
#include "Conversion/TCFToTCP/TCFToTCP.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
} // end namespace

void mlir::CANCER::registerConversionPasses() { ::registerPasses(); }
