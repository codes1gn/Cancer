// RUN: cancer-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @matmul_unknown
func @matmul_unknown(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_known
func @matmul_known(%arg0: tensor<27x16xf32>, %arg1: tensor<16x21xf32>) -> tensor<27x21xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  return %0 : tensor<27x21xf32>
}

// TODO check shape legal
// CHECK-LABEL: func @matmul_known_illegal
func @matmul_known_illegal(%arg0: tensor<27x18xf32>, %arg1: tensor<16x21xf32>) -> tensor<27x21xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<27x18xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<27x18xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  return %0 : tensor<27x21xf32>
}

