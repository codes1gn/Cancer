import mlir
from cancer_frontend.scaffold.mlir.custom_dialect import my_dialect as my_dialect

__all__ = [
    "NumpyRunner",
]


class NumpyRunner:
    """
    NumpyRunner acts like a compiler that support jit functionalities for numpy DSL.

    Returns:
        NumpyRunner: returns the instance of this class
    """

    def __init__(self):
        self.module = None

    def parse(self, code_path):
        self.module = mlir.parse_path(code_path, dialects=[my_dialect])

    def pretty_print(self):
        _str = self.module.pretty()
        print(_str)
