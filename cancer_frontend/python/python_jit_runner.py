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
MlirSsaId = astnodes.SsaId
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

    __slots__ = ["pass_manager"]

    def __init__(self):
        """
        Initializes the PythonRunner
        """

    def dump_mlir(self, _ast: MlirNode) -> str:
        dump_str = ""
        dump_str += "*******************&&&&&"
        dump_str += ColorPalette.FAIL
        dump_str += "\ndumping mlir ast\n"
        # dump_str += _ast.pretty()
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
        # dump_str += astunparse.dump(_ast)
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

    # TODO may change into classmethod or staticmethod
    def convert_python_to_mlir(self, pyast: ast.AST) -> MlirNode:
        """
        usage:
        self.pass_manager = PastToMlirPassManager()
        self.pass_manager.register_passes()
        return self.pass_manager.run(pyast)
        """
        from cancer_frontend.pass_manager import PastToMlirPassManager

        self.pass_manager = PastToMlirPassManager()
        self.pass_manager.register_passes()
        return astnodes.Block(label=None, body=[astnodes.FloatType(type=astnodes.FloatTypeEnum.f32)])
        # return self.pass_manager.run(pyast)


"""
        from mlir.dialects.standard import ReturnOperation
        from cancer_frontend.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp
        # this is return op
        #############################################################################
        # TODO understand what does match means
        _returnop = ReturnOperation(match=0)
        _ssaid_return = [MlirSsaId(value="0", op_no=None)]
        _returnop.values = _ssaid_return

        _rtype = [astnodes.FloatType(type=astnodes.FloatTypeEnum.f32)]
        _returnop.types = _rtype

        _returnop_wrapper = astnodes.Operation(result_list=None, op=_returnop, location=None)
        #############################################################################

        # this is tcf.add op
        #############################################################################
        _addop_result = [astnodes.OpResult(value=_ssaid_return[0], count=None)]
        _ssaid = [MlirSsaId(value="arg0", op_no=None)]
        _rtype = [
            astnodes.FunctionType(
                argument_types=[
                    astnodes.FloatType(type=astnodes.FloatTypeEnum.f32),
                    astnodes.FloatType(type=astnodes.FloatTypeEnum.f32),
                ],
                result_types=[astnodes.FloatType(type=astnodes.FloatTypeEnum.f32)],
            )
        ]
        _addop1 = TCF_AddOp(match=0, operand_a=_ssaid, operand_b=_ssaid, dtype=_rtype)
        _addop_wrapper = astnodes.Operation(result_list=_addop_result, op=_addop1, location=None)
        #############################################################################

        # this is block and region
        #############################################################################
        _block = astnodes.Block(label=None, body=[_addop_wrapper, _returnop_wrapper])
        _region = astnodes.Region(body=[_block])
        #############################################################################

        # this is function
        #############################################################################
        _name = astnodes.SymbolRefId(value="scalar_add_float")
        _args = [astnodes.NamedArgument(name=_ssaid, type=astnodes.FloatType(type=astnodes.FloatTypeEnum.f32))]
        _function = astnodes.Function(
            name=_name,
            args=_args,
            result_types=astnodes.FloatType(type=astnodes.FloatTypeEnum.f32),
            region=_region,
            attributes=None,
        )
        _function_wrapper = astnodes.Operation(result_list=[], op=_function, location=None)
        #############################################################################

        # this is outter module, out_block and out_region
        #############################################################################
        _out_block = astnodes.Block(label=None, body=[_function_wrapper])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=None, attributes=None, region=_out_region, location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])
        #############################################################################

        return _module

"""
