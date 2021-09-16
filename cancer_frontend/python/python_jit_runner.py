""" Contains jit runner class for compilation and execution """
import inspect
import textwrap
import ast
import astunparse
import copy
import sys

from typing import Callable
from imp import new_module
import traceback

import mlir
from mlir import astnodes as mlir_ast
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

    __slots__ = []

    def __init__(self):
        """
        Initializes the PythonRunner
        """

    def dump_mlir(self, _ast: mlir_ast.Node) -> str:
        dump_str = ""
        dump_str += "*******************&&&&&"
        dump_str += ColorPalette.FAIL
        dump_str += "\ndumping mlir ast\n"
        dump_str += str(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += ColorPalette.HEADER
        dump_str += "\ndumping mlir IR\n"
        dump_str += _ast.pretty()
        dump_str += ColorPalette.ENDC
        dump_str += "\n*******************&&&&&"
        return dump_str

    def dump_python(self, _ast: ast.AST) -> str:
        dump_str = ""
        dump_str += "*******************&&&&&"
        dump_str += ColorPalette.FAIL
        dump_str += "\ndumping python ast\n"
        dump_str += astunparse.dump(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += ColorPalette.HEADER
        dump_str += "\ndumping python code\n"
        dump_str += astunparse.unparse(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += "\n*******************&&&&&"
        return dump_str

    def parse_mlir(self, code_path: str) -> mlir_ast.Node:
        """
        Parses the code by providing its path
        :param
        """
        return mlir.parse_path(code_path, dialects=CANCER_DIALECTS)

    def parse_python(self, func: Callable) -> ast.AST:
        """
        Parses the code by providing its path
        :param
        """
        code_file = inspect.getsourcefile(func)
        code_lines, start_lineno = inspect.getsourcelines(func)
        code = "".join(code_lines)
        code = textwrap.dedent(code)
        pyast = ast.parse(code, filename=code_file)
        ast.increment_lineno(pyast, n=start_lineno - 1)
        return pyast
