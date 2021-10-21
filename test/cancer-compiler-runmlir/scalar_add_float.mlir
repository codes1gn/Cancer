// RUN: cancer-compiler-runmlir %s \
// RUN:   -invoke scalar_add_float \
// RUN:   -arg-value="1.0 : f32" \
// RUN:   -shared-libs=%cancer_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: 2.000000e+00 : f32
func @scalar_add_float(%arg0: f32) -> f32 {
  %0 =  tcf.add %arg0, %arg0 : (f32, f32) -> f32
  return %0 : f32
}

