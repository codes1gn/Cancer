""" Contains jit runner class for compilation and execution """

import mlir
from cancer_frontend.scaffold.mlir_dialects import *


__all__ = [
    "PythonRunner",
]


class PythonRunner:
    """
    PythonRunner class that is a compiler supports jit functionalities for numpy DSL.

    Returns:
        PythonRunner: returns the instance of this class
    """

    __slots__ = [
        "module",
        "source_code",
    ]

    def __init__(self):
        """
        Initializes the PythonRunner
        """
        self.module = None
        self.source_code = None

    def parse_mlir(self, code_path: str) -> None:
        """
        Parses the code by providing its path
        :param
        """
        self.module = mlir.parse_path(code_path, dialects=CANCER_DIALECTS)

    def pretty_mlir(self):
        print(self.module.pretty())
