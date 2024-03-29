// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke scalar \
// RUN:   -arg-value="dense<1.0> : tensor<f32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<f32>
func @scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = atir.add %arg0, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
