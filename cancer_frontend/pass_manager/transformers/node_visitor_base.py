import ast


class NodeVisitorBase(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
