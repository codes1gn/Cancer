// RUN: cancer-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @atir_add_test_mixed_arg_type
func @atir_add_test_mixed_arg_type(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) {
  // CHECK: atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %0 = atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  return
}

// CHECK-LABEL: func @atir_add_test
func @atir_add_test(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}


// CHECK-LABEL: func @atir_add_test_return
func @atir_add_test_return(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @atir_add_test_lhs_broadcast
// the broadcast is from scalar.
func @atir_add_test_lhs_broadcast(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK: atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %0 = atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// TODO casting element types
