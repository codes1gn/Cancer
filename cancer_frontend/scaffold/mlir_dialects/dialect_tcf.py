""" Implemented classes of Tensor Computation Flow Dialect. """

import inspect
import sys

from mlir import parse_string
from mlir.astnodes import Node, dump_or_value
from mlir.dialect import Dialect, DialectOp, DialectType, UnaryOperation, BinaryOperation, is_op


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


class TCF_MatmulOp(DialectOp):
    """AST node for an operation with an optional value."""

    _opname_ = "tcf.matmul"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "tcf.matmul {operand_a.ssa_use} , {operand_b.ssa_use} : {type.function_type}",
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
    ops=[m[1] for m in inspect.getmembers(sys.modules[__name__], lambda obj: is_op(obj, __name__))],
    types=[],
    preamble="",
    transformers=None,
)
