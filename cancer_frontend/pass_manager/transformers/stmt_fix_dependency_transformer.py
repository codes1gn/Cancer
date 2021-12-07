import ast
import astunparse
from astunparse.printer import Printer

from cancer_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase
from mlir.dialects.standard import ReturnOperation, ConstantOperation
from mlir import astnodes
from mlir.dialects.standard import *
from cancer_frontend.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId

__all__ = [
    "StmtFixDependencyTransformer",
]


class StmtFixDependencyTransformer(NodeTransformerBase):
    """This is fix dependcy Transformer defined in StmtNodeMappingTransformer class

    We map single python astnode to mlir astnode in StmtNodeMappingTransformer class,
    will consolidate all single node transformer to generate final mlir astnode.

    Attributtes:
        None.
    """

    __slots__ = []

    def __init__(self):
        """Initialize StmtFixDependencyTransformer class via inherit NodeTransformerBase.
        """

        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Method that constructs the FunctionDef in python_native dialect

        Args:
            node (ast.AST): python native astnode with mast_node attributions.

        Returns:
            ast.AST: python native astnode with mast_node attributions.
        """

        super().generic_visit(node)
        print(self.__str__(), "Fix Transformer::handling visit_FunctionDef on node\n")
        print("***** Python FunctionDef Node *****\n", astunparse.dump(node))
        
        print("***** MLIR Node fot FunctionDef *****\n",self.pretty_mlir(node.mast_node))
        
        # TODO Fix body elements in function region's block
        """ 
        Region: body (consist of a series Block)
        Need set ReturnOp Type according Assign op when Pass the Return assigned variable.
        """
        _blocks = node.mast_node.op.region.body
        operations = node.body
        # * obtain func argument type to List()
        argument_type = None
        if node.mast_node.op.args:
            argument_type = []
            for nameargument in node.mast_node.op.args:
                argument_type.append(nameargument.type)
        
        op_type = node.mast_node.op.result_types
        
        
        # if isinstance(operations[0].mast_node.op, ConstantOperation):
        #     op_type = operations[0].mast_node.op.type
        # if isinstance(operations[0].mast_node.op, ReturnOperation):
        #     op_type = operations[0].mast_node.op.types[0]
        """
        # if hasattr(operations[0].mast_node.op, "type"):
        #     op_type = operations[0].mast_node.op.type
        # if hasattr(operations[0].mast_node.op, "types") and isinstance(
        #         operations[0].mast_node.op.types, list):
        #     op_type = operations[0].mast_node.op.types[0]
        """
        for operation in operations:
            if isinstance(operation.mast_node.op, ReturnOperation):
                for i in range(len(operation.mast_node.op.types)):
                    operation.mast_node.op.types[i] = op_type
        print("len_blocks", len(_blocks))
        if operations:
            for i in range(len(_blocks)):
                _blocks[i].body.clear()
                for _, operation in enumerate(operations):
                    _blocks[i].body.append(operation.mast_node)

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Module in python_native dialect
        
        Module is the tops of python native ast, should traverses all nodes in Module,
        the final MLIR astnode is constructs according to the mast_node attribute of each node. 
        

        Args:
            node (ast.AST): python native astnode with all mast_node attributions.

        Returns:
            ast.AST: python native astnode with final MLIR astnode.
        """

        super().generic_visit(node)
        print(self.__str__(), "Fix handling visit_Module on node\n",
              astunparse.dump(node))

        for _module in node.mast_node.modules:
            for _block in _module.region.body:
                for index in range(len(_block.body)):
                    _block.body[index] = node.body[index].mast_node

        print(self.pretty_mlir(node.mast_node))

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """Method that constructs the ReturnOperation in python_native dialect

        Args:
            node (ast.AST): python native astnode with mast_node attributions.

        Returns:
            ast.AST: python native astnode with mast_node attributions.
        """

        super().generic_visit(node)
        print(self.__str__(), "Fix handling visit_Return on node\n",
              astunparse.dump(node))

        # fix returnop value
        # node.mast_node.op.values = node.value

        print(self.pretty_mlir(node.mast_node))

        return node

    # def visit_Name(self, node: ast.AST) -> ast.AST:
    #     """
    #     Method that constructs the ReturnOperation in python_native dialect
    #     """
    #     super().generic_visit(node)
    #     print(self.__str__(), "Fix handling visit_Name on node\n",
    #           astunparse.dump(node))

    #     return node
