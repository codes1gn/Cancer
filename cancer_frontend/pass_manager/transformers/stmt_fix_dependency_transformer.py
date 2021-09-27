import ast
import astunparse

from cancer_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase

from mlir import astnodes
from mlir.dialects.standard import ReturnOperation
from cancer_frontend.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId


__all__ = [
    "StmtFixDependencyTransformer",
]


class StmtFixDependencyTransformer(NodeTransformerBase):

    __slots__ = []

    def __init__(self):
        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """
        Method that constructs the FunctionDef in python_native dialect
        : param node,
        """
        super().generic_visit(node)
        print(self.__str__(), "handling visit_FunctionDef on node\n", astunparse.dump(node))

        # fix body elements in function region block
        _blocks = node.mast_node.op.region.body
        # TODO remove all hardcode in for loop
        for _block in _blocks:
            _block.body[0] = node.body[0].mast_node

        print(self.pretty_mlir(node.mast_node))

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        """
        Method that constructs the Module in python_native dialect
        """
        super().generic_visit(node)
        print(self.__str__(), "handling visit_Module on node\n", astunparse.dump(node))

        for _module in node.mast_node.modules:
            for _block in _module.region.body:
                for index in range(len(_block.body)):
                    _block.body[index] = node.body[index].mast_node

        print(self.pretty_mlir(node.mast_node))

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """
        Method that constructs the ReturnOperation in python_native dialect
        """
        super().generic_visit(node)
        print(self.__str__(), "handling visit_Return on node\n", astunparse.dump(node))

        # fix returnop value
        node.mast_node.op.values = node.value

        print(self.pretty_mlir(node.mast_node))

        return node
