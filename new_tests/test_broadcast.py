""" Tests pyMLIR on examples that use the Toy dialect. """

import os

from cancer_frontend.numpy_jit_runner import NumpyRunner


def test_broadcast():
    np_runner = NumpyRunner()
    np_runner.parse_mlir(os.path.join(os.path.dirname(__file__), "broadcast.mlir"))
    np_runner.pretty_mlir()


if __name__ == "__main__":
    test_broadcast()
