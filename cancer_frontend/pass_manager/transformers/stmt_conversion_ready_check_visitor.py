import ast
import astunparse

from cancer_frontend.scaffold.utils import *
from .node_visitor_base import NodeVisitorBase

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
    "StmtConversionReadyCheckTransformer",
]


class StmtConversionReadyCheckTransformer(NodeVisitorBase):

    __slots__ = []

    def __init__(self):
        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        assert node.mast_node is not None
        print("\n Check functiondef = \n", node.mast_node.pretty())

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        assert node.mast_node is not None
        print("\n Check module = \n", node.mast_node.pretty())

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        assert node.mast_node is not None
        print("\n Check return = \n", node.mast_node.pretty())

        return node
