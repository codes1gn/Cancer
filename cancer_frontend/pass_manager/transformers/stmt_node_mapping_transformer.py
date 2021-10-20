import ast
import astunparse
from astunparse import printer

from cancer_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase

from mlir import astnodes
from mlir.dialects.standard import ReturnOperation
from cancer_frontend.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId
MlirType = astnodes.FloatTypeEnum

__all__ = [
    "StmtNodeMappingTransformer",
]


class StmtNodeMappingTransformer(NodeTransformerBase):
    """This is class that mapping python ast stmt node to MLIR ast node.

    Transformer python ast stmt node to MLIR ast node 
    by setattr "mast_node" respectively.

    Attributes:
        None.
    """

    __slots__ = []

    def __init__(self):
        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Method that constructs the FunctionDef's corresponding MLIR node.

        Construct MLIR node by set FunctionDef attribute "mast_node".

        Args:
            node (ast.AST): FunctionDef node of python stmt.

        Returns:
            ast.AST: FunctionDef node with corresponding MLIR ast node.
        """
        super().generic_visit(node)
        print(self.__str__(), "handling visit_FunctionDef on node\n",
              astunparse.dump(node))

        _block = astnodes.Block(label=None, body=[None])
        _region = astnodes.Region(body=[_block])
        print("visit Functiondef:", type(node), " node name:\n", node.name)
        _name = astnodes.SymbolRefId(value=node.name)
        _args = None
        # _ssaid = [MlirSsaId(value="arg0", op_no=None)]
        # _args = [astnodes.NamedArgument(name=_ssaid, type=astnodes.FloatType(type=astnodes.FloatTypeEnum.f32))]

        _function = astnodes.Function(
            name=_name,
            args=_args,
            result_types=None,
            region=_region,
            attributes=None,
        )
        _function_wrapper = astnodes.Operation(result_list=[],
                                               op=_function,
                                               location=None)
        print(self.pretty_mlir(_function_wrapper))
        setattr(node, "mast_node", _function_wrapper)
        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Module's corresponding MLIR node.

        Construct MLIR node by set Module attribute "mast_node".

        Args:
            node (ast.AST): Module node of python ast.

        Returns:
            ast.AST: Module node with corresponding MLIR ast node.
        """
        super().generic_visit(node)
        print(self.__str__(), "handling visit_Module on node\n",
              astunparse.dump(node))

        _out_block = astnodes.Block(label=None, body=[None])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=None,
                                  attributes=None,
                                  region=_out_region,
                                  location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])

        print(self.pretty_mlir(_mlirfile))
        setattr(node, "mast_node", _mlirfile)
        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Return's corresponding MLIR node.

        Construct MLIR node by set Return attribute "mast_node".

        Args:
            node (ast.AST): Module node of python ast.

        Returns:
            ast.AST: Module node with corresponding MLIR ast node.
        """
        super().generic_visit(node)
        print(self.__str__(), "handling visit_Return on node\n",
              astunparse.dump(node))
        match = 1
        _returnop = ReturnOperation(match)
        _returnop.values = node.value
        if node.value:
            _values = list()
            _type = list()
            _values.append(MlirSsaId(value='ret' + str(match), op_no=None))
            if isinstance(node.value.value, float):
                _type.append(astnodes.FloatType(MlirType.f32))
            else:
                _type = None
            _returnop.values = _values
            _returnop.types = _type

        print("returnop dump:\n", _returnop.dump)
        _returnop_wrapper = astnodes.Operation(result_list=None,
                                               op=_returnop,
                                               location=None)
        print(self.pretty_mlir(_returnop_wrapper))
        setattr(node, "mast_node", _returnop_wrapper)
        return node

    # def visit_Name(self, node: ast.AST) -> ast.AST:
    #     """Method that constructs MLIR node via the func return type.

    #     Construct MLIR node by set Name attribute "mast_node".
    #     Name is return expr args.

    #     Args:
    #         node (ast.AST): Name node of python ast.

    #     Returns:
    #         ast.AST: Name node with corresponding MLIR ast node.
    #     """
    #     super().generic_visit(node)
    #     print(self.__str__(), "visit_Name on node\n", astunparse.dump(node))

    #     _type_wrapper = astnodes.FloatType(MlirType.f32)

    #     print("_type_wrapper:\n", self.pretty_mlir(_type_wrapper))
    #     setattr(node, "mast_node", _type_wrapper)
    #     return node
