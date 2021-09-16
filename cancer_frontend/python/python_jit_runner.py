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
from mlir import astnodes
from cancer_frontend.scaffold.mlir_dialects import *
from cancer_frontend.scaffold.utils import *


MlirNode = astnodes.Node
import types


def _pretty(self):
    result = self.dump_ast()
    lines = [""]
    indent = 0
    for index in range(len(result)):
        char = result[index]
        indent_word = "  "

        if char == " ":
            continue

        if char == "[" and result[index + 1] == "]":
            indent += 1
            lines[-1] += char
            continue

        if char == ",":
            lines[-1] += char
            lines.append(indent * indent_word)
            continue

        if char == "[":
            indent += 1
            lines[-1] += char
            lines.append(indent * indent_word)
            continue
        if char == "]":
            indent -= 1

        if char == "(":
            indent += 1
            lines[-1] += char
            lines.append(indent * "  ")
            continue
        if char == ")":
            indent -= 1

        if char != "\n":
            lines[-1] += char
        if char == "\n":
            lines.append(indent * indent_word)

    return "\n".join(lines)


MlirNode.pretty = _pretty


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

    def dump_mlir(self, _ast: MlirNode) -> str:
        dump_str = ""
        dump_str += "*******************&&&&&"
        dump_str += ColorPalette.FAIL
        dump_str += "\ndumping mlir ast\n"
        dump_str += _ast.pretty()
        dump_str += ColorPalette.ENDC
        dump_str += ColorPalette.HEADER
        dump_str += "\ndumping mlir IR\n"
        dump_str += _ast.dump()
        dump_str += ColorPalette.ENDC
        dump_str += "\n*******************&&&&&"
        return dump_str

    def dump_python(self, _ast: ast.AST) -> str:
        dump_str = ""
        dump_str += "*******************&&&&&"
        dump_str += ColorPalette.FAIL
        dump_str += "\ndumping python ast\n"
        # TODO use astunparse as alternative to ast pretty dump. this is not supported before py 3.8
        dump_str += astunparse.dump(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += ColorPalette.HEADER
        dump_str += "\ndumping python code\n"
        dump_str += astunparse.unparse(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += "\n*******************&&&&&"
        return dump_str

    def parse_mlir(self, code_path: str) -> MlirNode:
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
