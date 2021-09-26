import ast
from mlir import astnodes

MlirNode = astnodes.Node


class NodeTransformerBase(ast.NodeTransformer):
    def generic_visit(self, node):
        """
        printing visit messages
        """
        # self.fix_missing_parent(node)
        super().generic_visit(node)
        return node

    def pretty_mlir(self, node: MlirNode) -> str:
        result = node.dump_ast()
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

    """

    def fix_missing_parent(self, node):
        for child in ast.iter_child_nodes(node):
            setattr(child, "gemini_parent", node)
            assert hasattr(child, "gemini_parent")
    """
