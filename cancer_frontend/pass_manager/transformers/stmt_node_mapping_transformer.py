import ast
import astunparse
from astunparse import printer

from cancer_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase

from mlir import astnodes
from mlir.astnodes import CustomOperation, FunctionType, NamedArgument, Dimension
from mlir.dialects.standard import ReturnOperation, ConstantOperation
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
        """initialize the StatementConversionPass class.
        """
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
        print(self.__str__(), "Map transformer::handling visit_FunctionDef on node.\n")
        print(">>>>>Python FunctionDef Node:<<<<<\n", astunparse.dump(node))
        
        _block = astnodes.Block(label=None, body=[None])
        _region = astnodes.Region(body=[_block])
        _name = astnodes.SymbolRefId(value=node.name)
        _args = None
        if len(node.args.args) > 0:
            _args = []
            func_args = node.args.args
            for arg in func_args: 
                #TODO obtain args type
                if hasattr(arg.annotation, 'id') and arg.annotation.id == 'float':
                    _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None),
                                               type=astnodes.FloatType(MlirType.f32)))
                elif hasattr(arg.annotation, 'id') and arg.annotation.id == 'list':
                    # TODO：list -> <?xf32> <?x?xf32> <?x?x?f32> 
                    _type = astnodes.RankedTensorType(dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32))
                    _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None), type=_type))
                elif isinstance(arg.annotation, ast.Subscript) and isinstance(arg.annotation.slice, ast.Index) and arg.annotation.value.id == "List":
                    # TODO：List[float] -> <?xf32> <?x?xf32> <?x?x?f32> 
                    if arg.annotation.slice.value.id == 'float': # TODO: Other type, only support float now
                        _type = astnodes.RankedTensorType(dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32))
                        _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None), type=_type))
                else:
                    #TODO: Other type
                    pass
        _result_type = None
        if node.returns:
            if hasattr(node.returns, 'id') and node.returns.id == 'float':
                _result_type = astnodes.FloatType(MlirType.f32)
            elif hasattr(node.returns, 'id') and node.returns.id == 'list':
                _result_type = astnodes.RankedTensorType(dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32))
            elif isinstance(arg.annotation, ast.Subscript) and isinstance(arg.annotation.slice, ast.Index) and arg.annotation.value.id == "List":
                if arg.annotation.slice.value.id == 'float': # TODO: Other type, only support float now
                    _result_type = astnodes.RankedTensorType(dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32))
            else:
                #TODO: Other type
                pass
        
        _attributes = None

        _function = astnodes.Function(
            name=_name,
            args=_args,
            result_types=_result_type,
            region=_region,
            attributes=_attributes,
        )
        _function_wrapper = astnodes.Operation(result_list=[],
                                               op=_function,
                                               location=None)
        
        print("\n>>>>>MLIR Node for FunctionDef:<<<<<\n", self.pretty_mlir(_function_wrapper))
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
        print(self.__str__(), "Map transformer::handling visit_Module on node.\n")
        print(">>>>>Python Module Node:<<<<<\n", astunparse.dump(node))
        _name = None
        _attributes = None
        
        _out_block = astnodes.Block(label=None, body=[None])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=_name,
                                  attributes=_attributes,
                                  region=_out_region,
                                  location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])

        print("\n>>>>>MLIR Node for Module:<<<<<\n", self.pretty_mlir(_mlirfile))
        setattr(node, "mast_node", _mlirfile)

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Return corresponding MLIR node.

        Construct MLIR node by set Return attribute mast_node.

        Args:
            node (ast.AST): Module node of python ast.

        Returns:
            ast.AST: Module node with corresponding MLIR ast node.
        """

        super().generic_visit(node)
        print(self.__str__(), "Map transformer::handling visit_Return on node.\n")
        print(">>>>>Python Return Node:<<<<<\n", astunparse.dump(node))

        # if return value is not None, set match=1
        match = 0
        if node.value:
            match = 1
        _returnop = ReturnOperation(match)
        _returnop.values = node.value
        _values = list()
        _types = list()

        # two case of ReturnOp:
        # 1. return 1.0 -> return a specific value
        # 2. return var -> return the value specified by a variable
        if isinstance(node.value, ast.Constant):
            _value = 'ret' + str(match)
            _op_no = None
            
            _values.append(MlirSsaId(value=_value, op_no=_op_no))
            if isinstance(node.value.value, float):
                _types.append(astnodes.FloatType(MlirType.f32))
            else:
                _types = None
            _returnop.values = _values
            _returnop.types = _types

        if isinstance(node.value, ast.Name):
            _value = node.value.id
            _op_no = None
            
            _values.append(MlirSsaId(value=_value, op_no=_op_no))
            _types.append(None)
            _returnop.values = _values
            _returnop.types = _types

        _returnop_wrapper = astnodes.Operation(result_list=None,
                                               op=_returnop,
                                               location=None)
        print("\nMLIR Node for Return:<<<<<\n", self.pretty_mlir(_returnop_wrapper))
        setattr(node, "mast_node", _returnop_wrapper)

        return node

    def visit_Assign(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Assign  corresponding MLIR node.

        Construct MLIR node by set Assign astnode  attribute mast_node.

        Args:
            node (ast.AST): Assign astnode of python

        Returns:
            ast.AST: Assign astnode of python with mast_node attributions.
        """
        
        super().generic_visit(node)
        print(self.__str__(), "Map transformer::handling visit_Assign on node.\n")
        print(">>>>>Python Assign Node:<<<<<\n",astunparse.dump(node))
        
        _type = None
        if isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, float):
                _match = 0
                _value = node.value.value
                
                # _type = astnodes.FloatType(MlirType.f32)
                _type = None
            
                _assignop = ConstantOperation(match=_match, value=_value, type=_type)
                
                _result_list = list()
                
                _value = node.targets[0].id
                _SsaId = MlirSsaId(value=_value, op_no=None)
                _result_list.append(astnodes.OpResult(value=_SsaId, count=None))

                _assignop_wrapper = astnodes.Operation(result_list=_result_list,
                                                    op=_assignop,
                                                    location=None)
                print(">>>>>MLIR Node for Assign:<<<<<\n", self.pretty_mlir(_assignop_wrapper))
                setattr(node, "mast_node", _assignop_wrapper)

        if isinstance(node.value, ast.BinOp):
            _namespace = 'tcf'
            _name = None
            if isinstance(node.value.op, ast.Add):
                _name = 'add'

            _args = list()
            _SsaId_left = MlirSsaId(value=node.value.left.id, op_no=None)
            _SsaId_right = MlirSsaId(value=node.value.right.id, op_no=None)
            _args.extend([_SsaId_left, _SsaId_right])
            
            _argument_types = [_type, _type]
            _result_types = [_type]
            _type_binop = FunctionType(argument_types=_argument_types, result_types=_result_types)
            
            _assignop = CustomOperation(namespace=_namespace, name=_name, args=_args, type=_type_binop)
            
            _result_list = list()
            _result_list.append(astnodes.OpResult(value=MlirSsaId(value=node.targets[0].id, op_no=None), count=None))
            _assignop_wrapper = astnodes.Operation(result_list=_result_list,
                                                    op=_assignop,
                                                    location=None)
            print(">>>>>MLIR Node for Assign BinOp:<<<<<\n", self.pretty_mlir(_assignop_wrapper))
            setattr(node, "mast_node", _assignop_wrapper)
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