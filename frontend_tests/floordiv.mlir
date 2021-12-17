func @floordiv_scalar0(%arg0: f32) -> f32 {
    %var0 = constant 2.0 : f32
    %res = tcf.floordiv %arg0, %var0 : (f32, f32) -> f32
    return %res : f32
}

func @floordiv_scalar1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.floordiv %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @floordiv_scalar2 -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.floordiv %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @floordiv_scalar3(%arg0: f32, %arg1: f32) -> f32 {
    %arg0 = tcf.floordiv %arg0, %arg0 : (f32, f32) -> f32
    return %arg0 : f32
}

func @listfloordiv0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.floordiv %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}

func @listfloordiv1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.floordiv %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}