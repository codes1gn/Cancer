func @scalar_add0(%arg0: f32) -> f32 {
  %var0 = constant 2.0 : f32
  %res =  tcf.add %arg0, %var0 : (f32, f32) -> f32
  return %res : f32
}