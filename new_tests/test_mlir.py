""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from cancer_frontend.numpy_jit_runner import NumpyRunner

MLIR_INPUTS = [
    "basic.mlir",
    "broadcast.mlir",
    # "constant-add.mlir",
    # "constant-add-scalar.mlir",
    # "constant.mlir",
    # "control-flow-basic.mlir",
    "conv_2d_nchw.mlir",
    "elementwise.mlir",
    "identity.mlir",
    # "invalid-broadcast.mlir",
    # "invalid-conv_2d_nchw.mlir",
    # "invalid-matmul.mlir",
    # "invalid-num-inputs.mlir",
    "matmul.mlir",
    "mixed-rank.mlir",
    "scalar.mlir",
    "multi-output.mlir",
    "multiple-ops.mlir",
    # "pad.mlir",
]


def test_main():
    np_runner = NumpyRunner()
    for input_file in MLIR_INPUTS:
        print("*****************************************************************")
        print("testing " + input_file + "\n")
        np_runner.parse_mlir(os.path.join(os.path.dirname(__file__), input_file))
        np_runner.pretty_mlir()
        print("")
        print("*****************************************************************")
        print("")
        print("")


if __name__ == "__main__":
    test_main()
