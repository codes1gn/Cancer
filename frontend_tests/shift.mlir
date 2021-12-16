func @lshift_scalar0(%arg0: f32) -> f32 {
    %var0 = constant 2.0 : f32
    %res = tcf.lshift %arg0, %var0 : (f32, f32) -> f32
    return %res : f32
}

func @lshift_scalar1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.lshift %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @lshift_scalar2 -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.lshift %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @lshift_scalar3(%arg0: f32, %arg1: f32) -> f32 {
    %arg0 = tcf.lshift %arg0, %arg1 : (f32, f32) -> f32
    return %arg0 : f32
}

func @listLshift0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.lshift %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}

func @listLshift1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.lshift %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}


func @rshift_scalar0(%arg0: f32) -> f32 {
    %var0 = constant 2.0 : f32
    %res = tcf.rshift %arg0, %var0 : (f32, f32) -> f32
    return %res : f32
}

func @rshift_scalar1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.rshift %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @rshift_scalar2 -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.rshift %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @rshift_scalar3(%arg0: f32, %arg1: f32) -> f32 {
    %arg0 = tcf.rshift %arg0, %arg0 : (f32, f32) -> f32
    return %arg0 : f32
}


func @listRshift0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.rshift %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}

func @listRshift1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.rshift %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
}
