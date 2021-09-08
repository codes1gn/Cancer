""" Contains jit runner class for compilation and execution """

import mlir
from cancer_frontend.scaffold.mlir import *

__all__ = [
    "NumpyRunner",
]


class NumpyRunner:
    """
    NumpyRunner class that is a compiler supports jit functionalities for numpy DSL.

    Returns:
        NumpyRunner: returns the instance of this class
    """

    __slots__ = [
        "module",
        "source_code",
    ]

    def __init__(self):
        """
        Initializes the NumpyRunner
        """
        self.module = None
        self.source_code = None

    def parse_mlir(self, code_path: str) -> None:
        """
        Parses the code by providing its path
        # TODO change in parse_mlir
        """
        self.module = mlir.parse_path(code_path, dialects=[my_dialect])

    def pretty_mlir(self):
        print(self.module.pretty())
