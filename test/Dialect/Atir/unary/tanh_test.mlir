// RUN: cancer-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @atir_tanh_test
func @atir_tanh_test(%arg0: tensor<?xf32>) {
  // CHECK: atir.tanh %arg0 : tensor<?xf32>
  %0 = atir.tanh %arg0: tensor<?xf32>
  return
}

// CHECK-LABEL: func @atir_tanh_test_return
func @atir_tanh_test_return(%arg0: tensor<?xf32>) -> tensor<?xf32>{
  // CHECK: atir.tanh %arg0 : tensor<?xf32>
  %0 = atir.tanh %arg0: tensor<?xf32>
  return %0 : tensor<?xf32>
}

