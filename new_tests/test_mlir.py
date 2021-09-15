""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from cancer_frontend.python import PythonRunner

MLIR_INPUTS = [
    "constant.mlir",
    "identity.mlir",
    "basic.mlir",
    "constant-add.mlir",
    # "constant-add-scalar.mlir",
    # "control-flow-basic.mlir",
    # "broadcast.mlir",
    # "conv_2d_nchw.mlir",
    # "elementwise.mlir",
    # "invalid-broadcast.mlir",
    # "invalid-conv_2d_nchw.mlir",
    # "invalid-matmul.mlir",
    # "invalid-num-inputs.mlir",
    # "matmul.mlir",
    # "mixed-rank.mlir",
    # "scalar.mlir",
    # "multi-output.mlir",
    # "multiple-ops.mlir",
    # "pad.mlir",
]


def test_main():
    py_runner = PythonRunner()
    for input_file in MLIR_INPUTS:
        print("*****************************************************************")
        print("testing " + input_file + "\n")
        py_runner.parse_mlir(os.path.join(os.path.dirname(__file__), input_file))
        py_runner.dump_module()
        print("")
        print("*****************************************************************")
        print("")
        print("")


if __name__ == "__main__":
    test_main()
