""" Contains jit runner class for compilation and execution """

import mlir
from cancer_frontend.scaffold.mlir_dialects import *
from cancer_frontend.scaffold.utils import *


__all__ = [
    "PythonRunner",
]


class PythonRunner:
    """
    PythonRunner class that is a compiler supports jit functionalities for numpy DSL.

    Returns:
        PythonRunner: returns the instance of this class
    """

    __slots__ = ["mlir_ast", "mlir_code", "python_code", "python_ast"]

    def __init__(self):
        """
        Initializes the PythonRunner
        """
        self.mlir_ast = None
        self.python_ast = None
        self.mlir_code = None
        self.python_code = None

    def parse_mlir(self, code_path: str) -> None:
        """
        Parses the code by providing its path
        :param
        """
        self.mlir_ast = mlir.parse_path(code_path, dialects=CANCER_DIALECTS)

    def dump_mlir(self):
        print("*******************&&&&&")
        print("dumping mlir ast")
        print(self.mlir_ast)
        print("dumping mlir IR")
        print(self.mlir_ast.pretty())
        print("*******************&&&&&")

    def parse_python(self, code_path: str) -> None:
        """
        Parses the code by providing its path
        :param
        """
        src_code = read_src(code_path)
        print(src_code)
        assert 0
