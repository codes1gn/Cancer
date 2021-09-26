import ast
import astunparse

from cancer_frontend.scaffold.utils import *
from .node_visitor_base import NodeVisitorBase

from mlir import astnodes
from mlir.dialects.standard import ReturnOperation
from cancer_frontend.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId


__all__ = [
    "StmtConversionReadyCheckVisitor",
]


class StmtConversionReadyCheckVisitor(NodeVisitorBase):

    __slots__ = []

    def __init__(self):
        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        assert node.mast_node is not None
        print("\n Check FunctionDef = \n", self.pretty_mlir(node.mast_node))

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        assert node.mast_node is not None
        print("\n Check Module = \n", self.pretty_mlir(node.mast_node))

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        assert node.mast_node is not None
        print("\n Check ReturnOp = \n", self.pretty_mlir(node.mast_node))

        return node
