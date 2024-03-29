// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<2x1x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BATCH

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<2x2x1x1xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME_CHANNELS

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x1x1xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT_CHANNELS

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TINY_SQUARE

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x32x32xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x32x32xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HUGE_SQUARE

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x0x0xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZERO_KH_KW

// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke conv_2d_cfirst \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x0x0xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x0x0xf32>" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZERO_H_W

// BATCH: output #0: dense<0.000000e+00> : tensor<2x1x1x1xf32>

// SAME_CHANNELS: output #0: dense<0.000000e+00> : tensor<1x2x1x1xf32>

// DIFFERENT_CHANNELS: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

// TINY_SQUARE: output #0: dense<0.000000e+00> : tensor<1x1x2x2xf32>

// HUGE_SQUARE: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

// ZERO_KH_KW: output #0: dense<0.000000e+00> : tensor<1x1x3x3xf32>

// ZERO_H_W: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

func @conv_2d_cfirst(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
