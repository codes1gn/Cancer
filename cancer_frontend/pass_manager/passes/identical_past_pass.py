import ast

from cancer_frontend.pass_manager.transformers import *
from .pass_base import PassBase

__all__ = [
    "IdenticalPastPass",
]


class IdenticalPastPass(PassBase):

    __slots__ = [
        "solvers",
    ]

    def __init__(self):
        # type: (None) -> None
        super(IdenticalPastPass, self).__init__()
        self.solvers = []
        self.solvers.append(IdenticalPastTransformer)

    def run_pass(self, ast_root: ast.AST) -> ast.AST:
        for _solver in self.solvers:
            ast_root = _solver().visit(ast_root)
            ast.fix_missing_locations(ast_root)

        return ast_root
