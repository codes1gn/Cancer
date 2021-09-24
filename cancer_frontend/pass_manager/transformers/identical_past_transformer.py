import ast

from cancer_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    "IdenticalPastTransformer",
]


class IdenticalPastTransformer(NodeTransformerBase):

    __slots__ = []

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_Call(self, node: ast.AST) -> ast.AST:
        return node
