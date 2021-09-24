import ast
import astunparse

from cancer_frontend.pass_manager.transformers import *
from cancer_frontend.pass_manager.passes.pass_base import PassBase

__all__ = [
    "StatementConversionPass",
]


class StatementConversionPass(PassBase):

    __slots__ = [
        "solvers",
    ]

    def __init__(self):
        # type: (None) -> None
        super().__init__()
        self.solvers = []
        self.solvers.append(ReturnOpTransformer)

    def run_pass(self, ast_root: ast.AST) -> ast.AST:
        for _solver in self.solvers:
            ast_root = _solver().visit(ast_root)
            print(ast_root.body[0].body[0].mast_node.pretty())
            print(ast_root.body[0].mast_node.pretty())
            print(ast_root.mast_node.pretty())
            ast.fix_missing_locations(ast_root)

        return ast_root
