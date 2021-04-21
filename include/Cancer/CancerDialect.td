//===- CancerDialect.td - Cancer dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_DIALECT
#define CANCER_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cancer dialect definition.
//===----------------------------------------------------------------------===//

def Cancer_Dialect : Dialect {
    let name = "cancer";
    let summary = "A cancer out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::cancer";
}

//===----------------------------------------------------------------------===//
// Base cancer operation definition.
//===----------------------------------------------------------------------===//

class Cancer_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Cancer_Dialect, mnemonic, traits>;

#endif // CANCER_DIALECT
