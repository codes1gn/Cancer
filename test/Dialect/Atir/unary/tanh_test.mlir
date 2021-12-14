// RUN: cancer-opt <%s | FileCheck %s --dump-input=fail

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

// CHECK-LABEL: func @atir_max_test
func @atir_max_test(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: atir.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: atir.exp %arg0 : tensor<?xf32>
  %1 = atir.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = atir.exp %arg0 : tensor<?xf32>
  return
}

// CHECK-LABEL: func @atir_max_test
func @atir_max_test_return(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: atir.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = atir.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func @matmul
func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
