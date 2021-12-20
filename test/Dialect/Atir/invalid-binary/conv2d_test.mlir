// RUN: not cancer-opt <%s | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @conv_2d_illegal_unknown
func @conv_2d_illegal_unknown(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-FAIL: atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
