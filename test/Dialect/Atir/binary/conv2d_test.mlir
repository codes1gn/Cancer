// RUN: cancer-opt <%s | FileCheck %s


// CHECK-LABEL: func @conv_2d_channel_first
func @conv_2d_channel_first(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @conv_2d_channel_last
func @conv_2d_channel_last(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: atir.conv_2d_clast %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = atir.conv_2d_clast %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
