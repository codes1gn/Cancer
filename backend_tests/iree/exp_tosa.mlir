module  {
  func @atir_exp_test_scalar_noret(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = "tosa.exp"(%arg0) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

