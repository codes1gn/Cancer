""" Dialect classes that create and custom dialect as example. """

from mlir import parse_string
from mlir.astnodes import Node, dump_or_value
from mlir.dialect import Dialect, DialectOp, DialectType


##############################################################################
# Dialect Types

__all__ = [
    "DIALECT_PYNATIVE",
]


##############################################################################
# Dialect Operations


class ConstantOper(DialectOp):
    """AST node for an operation with an optional value."""

    _syntax_ = [
        "pynative.constant {arg.ssa_id} : {type.tensor_type}",
        "pynative.constant {arg.constant_literal} : {type.tensor_type}",
        "pynative.constant {arg.float_literal} : {type.tensor_type}",
    ]


##############################################################################
# Dialect

DIALECT_PYNATIVE = Dialect(
    "pynative",
    ops=[ConstantOper],
    types=[],
    preamble="",
    transformers=None,
)
