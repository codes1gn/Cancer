""" Dialect classes that create and custom dialect as example. """

from mlir import parse_string
from mlir.astnodes import Node, dump_or_value
from mlir.dialect import Dialect, DialectOp, DialectType, BinaryOperation


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

    # TODO in syntax, between string_literals and non-terminals, must be seperated with whitespace
    _syntax_ = [
        "tcf.add {operand_a.ssa_use} , {operand_b.ssa_use} : {type.type}",
    ]


# class TCF_AddOp(BinaryOperation):
#    _opname_ = "tcf.add"


##############################################################################
# Dialect

DIALECT_TCF = Dialect(
    "tcf",
    ops=[TCF_AddOp],
    types=[],
    preamble="",
    transformers=None,
)
