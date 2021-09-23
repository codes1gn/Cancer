import ast


from ..transformer.plugin_multiply_transformer import PluginMultiplyTransformer
from .pass_base import PassBase

__all__ = [
    "PluginMultiplyPass",
]


class PluginMultiplyPass(PassBase):

    __slots__ = [
        "_solvers",
    ]

    def __init__(self):
        # type: (None) -> None
        super(PluginMultiplyPass, self).__init__()
        self._solvers = []
        self._solvers.append(PluginMultiplyTransformer)

    def run_pass(self, ast_root: ast.AST) -> ast.AST:
        solver1 = self._solvers[0]()

        ast_root = solver1.visit(ast_root)
        ast.fix_missing_locations(ast_root)

        return ast_root
