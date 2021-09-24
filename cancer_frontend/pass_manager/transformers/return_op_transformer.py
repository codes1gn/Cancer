import ast
import astunparse

from cancer_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase

from mlir import astnodes
from mlir.dialects.standard import ReturnOperation
from cancer_frontend.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId


def _pretty(self: MlirNode) -> str:
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
    "ReturnOpTransformer",
]


class ReturnOpTransformer(NodeTransformerBase):

    __slots__ = []

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        print("\nhandling functiondef = ", ast.dump(node))

        _block = astnodes.Block(label=None, body=[None])
        _region = astnodes.Region(body=[_block])

        _name = astnodes.SymbolRefId(value="constant1")

        _args = None
        # _ssaid = [MlirSsaId(value="arg0", op_no=None)]
        # _args = [astnodes.NamedArgument(name=_ssaid, type=astnodes.FloatType(type=astnodes.FloatTypeEnum.f32))]
        _function = astnodes.Function(
            name=_name,
            args=_args,
            result_types=None,
            region=_region,
            attributes=None,
        )
        _function_wrapper = astnodes.Operation(result_list=[], op=_function, location=None)
        print(_function_wrapper.pretty())
        setattr(node, "mast_node", _function_wrapper)

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        print("\nhandling module = ", ast.dump(node))

        _out_block = astnodes.Block(label=None, body=[None])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=None, attributes=None, region=_out_region, location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])

        print(_mlirfile.pretty())
        setattr(node, "mast_node", _mlirfile)

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)

        _returnop = ReturnOperation(match=0)
        _returnop.values = node.value
        _returnop.types = None
        _returnop_wrapper = astnodes.Operation(result_list=None, op=_returnop, location=None)
        print(_returnop_wrapper.pretty())
        setattr(node, "mast_node", _returnop_wrapper)

        return node
