// RUN: not cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x2x2xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHANNELS

// RUN: not cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x3x2xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HEIGHT

// RUN: not cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x3xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIDTH

// CHANNELS: CANCER: aborting: input and filter in-channels must be equal
// HEIGHT: CANCER: aborting: input height must be greater than or equal to filter KH-dimension
// WIDTH: CANCER: aborting: input width must be greater than or equal to filter KW-dimension
func @conv_2d_nchw(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = atir.conv_2d_nchw %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
