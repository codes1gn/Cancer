func @scalar_add2() -> f32 {
    %var0 = constant 1.0 : f32
    %var1 = constant 2.0 : f32
    %res = tcf.add %var0, %var1 : (f32, f32) -> f32
    return %res : f32
}