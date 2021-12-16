func @bitor_scalar0(%arg0: f32) -> f32 {
    %var0 = constant 2.0 : f32
    %res = tcf.bitor %arg0, %var0 : (f32, f32) -> f32
    return %res : f32
}

func @bitor_scalar1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.bitor %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @bitor_scalar2 -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.bitor %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @bitor_scalar3(%arg0: f32, %arg1: f32) -> f32 {
    %arg0 = tcf.bitor %arg0, %arg0 : (f32, f32) -> f32
    return %arg0 : f32
}

func @listbitor0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.bitor %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}

func @listbitor1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.bitor %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}
