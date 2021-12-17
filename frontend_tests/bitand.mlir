func @bitand_scalar0(%arg0: f32) -> f32 {
    %var0 = constant 2.0 : f32
    %res = tcf.bitand %arg0, %var0 : (f32, f32) -> f32
    return %res : f32
}

func @bitand_scalar1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.bitand %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}

func @bitand_scalar2 -> f32 {
    %var0 = constant 1.0 : f32  
    %var1 = constant 2.0 : f32
    %res = tcf.bitand %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}

func @bitand_scalar3(%arg0: f32, %arg1: f32) -> f32 {    
    %arg0 = tcf.bitand %arg0, %arg0 : (f32, f32) -> f32    
    return %arg0 : f32                                     
}

func @listbitand0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> { 
    %res = tcf.bitand %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32> 
    return %res : tensor<?x?xf32>                          
}

func @listbitand1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %res = tcf.bitand %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32> 
}