import mlir
from cancer_frontend.scaffold.mlir.custom_dialect import my_dialect as my_dialect


##############################################################################
# Tests


def test_custom_dialect():
    code = """module {
  func @toy_test(%ragged: !toy.ragged<coo+csr, 32x14xf64>) -> tensor<32x14xf64> {
    %t_tensor = toy.densify %ragged : tensor<32x14xf64>
    return %t_tensor : tensor<32x14xf64>
  }
}"""
    m = mlir.parse_string(code, dialects=[my_dialect])
    dump = m.pretty()
    print(dump)

    # Test for round-trip
    assert dump == code


if __name__ == "__main__":
    test_custom_dialect()
