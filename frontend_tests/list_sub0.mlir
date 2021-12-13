func @list_add0(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %res = tcf.sub %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %res : tensor<?xf32>
}
