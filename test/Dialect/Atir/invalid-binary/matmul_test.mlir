// RUN: not cancer-opt <%s | FileCheck %s --dump-input=fail

// TODO check shape illegal
// CHECK-LABEL: func @matmul_illegal_known
func @matmul_illegal_known(%arg0: tensor<27x16x1xf32>, %arg1: tensor<16x21x1xf32>) -> tensor<27x21x1xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<27x16x1xf32>, tensor<16x21x1xf32>) -> tensor<27x21x1xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<27x16x1xf32>, tensor<16x21x1xf32>) -> tensor<27x21x1xf32>
  return %0 : tensor<27x21x1xf32>
}

