import iree.compiler as ireecc
import iree.runtime as ireert

import numpy as np

exp_scalar = """
module {
  func @exp_scalar(%arg0 : tensor<f32>) -> tensor<f32> {
    %0 = "tosa.exp"(%arg0) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
"""

mul_tosa = """
func @mul_tosa(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
"""

add_tosa = """
func @add_tosa(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
"""

scalar_add_tosa = """
func @scalar_add_tosa(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
"""

class IREEInvoker:
    def __init__(self, iree_module):
        self._iree_module = iree_module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            return self._iree_module[function_name](*args)
        return invoke

# test 1: test compile str of mul_tosa and run on dylib
binary_mul_dylib_from_str = ireecc.tools.compile_str(
        mul_tosa,
        input_type="tosa",
        target_backends=["dylib"]
        )

# test 2: test compile str of add_tosa and run on vulkan
binary_add_vulkan_from_str = ireecc.tools.compile_str(
        scalar_add_tosa,
        input_type="tosa",
        target_backends=["vulkan-spirv"]
        )

# test 2: test compile str of add_tosa and run on vulkan
binary_exp_vulkan_from_str = ireecc.tools.compile_str(
        exp_scalar,
        input_type="tosa",
        target_backends=["vulkan-spirv"]
        )

# way 1 system api, higher than raw python bindings
## vm_module = ireert.VmModule.from_flatbuffer(binary_mul_dylib_from_str)
# config = ireert.Config(driver_name="dylib")
# ctx = ireert.SystemContext(config=config)
# ctx.add_vm_module(vm_module)
# _callable = ctx.modules.module["mul_tosa"]
# arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
# arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
# result = _callable(arg0, arg1)
# np.testing.assert_allclose(result, [4., 10., 18., 28.])
# print("result: ", result)
vm_module = ireert.VmModule.from_flatbuffer(binary_exp_vulkan_from_str)
config = ireert.Config(driver_name="vulkan")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
_callable = ctx.modules.module["exp_scalar"]
arg0 = np.array(1., dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
result = _callable(arg0)
np.testing.assert_allclose(result, [2.718282])
print("result: ", result)

# TODO add this (albert): way 2 low-level capi bindings to python

