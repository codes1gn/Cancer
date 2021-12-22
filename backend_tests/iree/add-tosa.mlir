func @tensor_float(%0 : tensor<4xf32>, %1 : tensor<4xf32>) -> tensor<4xf32> {
  %result = "tosa.add"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

