""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from cancer_frontend.numpy_jit_runner import NumpyRunner


def test_basic():
    np_runner = NumpyRunner()
    np_runner.parse_mlir(os.path.join(os.path.dirname(__file__), "basic.mlir"))
    np_runner.pretty_mlir()


if __name__ == "__main__":
    test_basic()
