""" Tests pyMLIR on examples that use the Toy dialect. """

from mlir import parse_string, parse_path
import os


def test_constant():
    module = parse_path(os.path.join(os.path.dirname(__file__), "constant.mlir"))
    print(module.pretty())


if __name__ == "__main__":
    test_constant()
