""" Dialect classes that create and custom dialect as example. """

from mlir import parse_string
from mlir.astnodes import Node, dump_or_value
from mlir.dialect import Dialect, DialectOp, DialectType, UnaryOperation, BinaryOperation


##############################################################################
# Dialect Types

__all__ = [
    "DIALECT_TCF",
]


##############################################################################
# Dialect Operations


class TCF_AddOp(DialectOp):
    """AST node for an operation with an optional value."""

    _opname_ = "tcf.add"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "tcf.add {operand_a.ssa_use} , {operand_b.ssa_use} : {type.function_type}",
    ]


class TCF_MaxOp(DialectOp):
    """AST node for an operation with an optional value."""

    _opname_ = "tcf.max"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "tcf.max {operand_a.ssa_use} , {operand_b.ssa_use} : {type.function_type}",
    ]


class TCF_MulOp(DialectOp):
    """AST node for an operation with an optional value."""

    _opname_ = "tcf.mul"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "tcf.mul {operand_a.ssa_use} , {operand_b.ssa_use} : {type.function_type}",
    ]


class TCF_Conv2DChannelFirstOp(DialectOp):
    """AST node for an operation with an optional value."""

    _opname_ = "tcf.conv_2d_nchw"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "tcf.conv_2d_nchw {operand_a.ssa_use} , {operand_b.ssa_use} : {type.function_type}",
    ]


class TCF_Conv2DChannelLastOp(DialectOp):
    """AST node for an operation with an optional value."""

    _opname_ = "tcf.conv_2d_nhwc"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "tcf.conv_2d_nhwc {operand_a.ssa_use} , {operand_b.ssa_use} : {type.function_type}",
    ]


class TCF_TanhOp(UnaryOperation):
    _opname_ = "tcf.tanh"


class TCF_ExpOp(UnaryOperation):
    _opname_ = "tcf.exp"


##############################################################################
# Dialect

DIALECT_TCF = Dialect(
    "tcf",
    ops=[TCF_AddOp, TCF_MulOp, TCF_MaxOp, TCF_ExpOp, TCF_TanhOp, TCF_Conv2DChannelLastOp, TCF_Conv2DChannelFirstOp],
    types=[],
    preamble="",
    transformers=None,
)
