//===- CancerDialect.td - Cancer dialect -----------*- tablegen -*-===//

#ifndef CANCER_DIALECT
#define CANCER_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cancer dialect definition.
//===----------------------------------------------------------------------===//

def Cancer_Dialect : Dialect {
    let name = "cancer";
    let summary = "Cancer Dialect for Common Acceleration Computation Representation.";
    let description = [{
        This dialect is for Common Acceleration Computation Representation, which consumes numpy and pytorch programs as inputs, and make use of MLIR infrastructures and compiles with llvm codegen support.
    }];
    let cppNamespace = "::mlir::cancer";
}

//===----------------------------------------------------------------------===//
// Base cancer operation definition.
//===----------------------------------------------------------------------===//

class Cancer_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Cancer_Dialect, mnemonic, traits>;

#endif // CANCER_DIALECT
