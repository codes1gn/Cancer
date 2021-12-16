func @bitxor_scalar0(%arg0: f32) -> f32 {
    %var0 = constant 2.0 : f32
    %res = tcf.bitxor %arg0, %var0 : (f32, f32) -> f32
    return %res : f32
}

func @bitxor_scalar1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.bitxor %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @bitxor_scalar2 -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.bitxor %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @bitxor_scalar3(%arg0: f32, %arg1: f32) -> f32 {
    %arg0 = tcf.bitxor %arg0, %arg0 : (f32, f32) -> f32
    return %arg0 : f32
}

func @listbitxor0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.bitxor %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}

func @listbitxor1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.bitxor %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }


