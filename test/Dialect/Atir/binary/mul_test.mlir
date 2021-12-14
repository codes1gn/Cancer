// RUN: cancer-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @atir_mul_test
func @atir_mul_test(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}


// CHECK-LABEL: func @atir_mul_test_return
func @atir_mul_test_return(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @atir_mul_test_lhs_broadcast
// the broadcast is from scalar.
func @atir_mul_test_lhs_broadcast(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK: atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// TODO casting element types
