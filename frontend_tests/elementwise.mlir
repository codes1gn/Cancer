// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke max \
// RUN:   -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -arg-value="dense<[3.0]> : tensor<1xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MAX

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke mul \
// RUN:   -arg-value="dense<[1.0, 2.0]> : tensor<2xf32>" \
// RUN:   -arg-value="dense<[3.0, 4.0]> : tensor<2xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MUL

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke exp \
// RUN:   -arg-value="dense<[0.0, 1.0]> : tensor<2xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXP

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke tanh \
// RUN:   -arg-value="dense<[0.0, 1.0]> : tensor<2xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TANH

// These ops share a lot of code paths. So we don't test the exact
// broadcasting behavior and error checking for all of them.

// MAX: output #0: dense<3.000000e+00> : tensor<1xf32>
func @max(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 =  tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// MUL: output #0: dense<[3.000000e+00, 8.000000e+00]> : tensor<2xf32>
func @mul(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 =  tcf.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// EXP: output #0: dense<[1.000000e+00, 2.71828175]> : tensor<2xf32>
func @exp(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcf.exp %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// TANH: output #0: dense<[0.000000e+00, 0.761594116]> : tensor<2xf32>
func @tanh(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcf.tanh %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}
