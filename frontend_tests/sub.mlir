func @scalar_add0(%arg0: f32) -> f32 {
  %var0 = constant 2.0 : f32
  %res =  tcf.sub %arg0, %var0 : (f32, f32) -> f32
  return %res : f32
}

func @scalar_add1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.sub %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @scalar_add2() -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.sub %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @list_add0(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %res = tcf.sub %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %res : tensor<?xf32>
}
