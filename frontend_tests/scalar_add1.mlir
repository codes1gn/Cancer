func @scalar_add1(%arg0: f32, %arg1: f32) -> f32 {
    %res = tcf.add %arg0, %arg1 : (f32, f32) -> f32
    return %res : f32
}