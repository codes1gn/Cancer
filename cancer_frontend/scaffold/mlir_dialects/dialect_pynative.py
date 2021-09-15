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


class PYNATIVE_ConstantOp(DialectOp):
    """AST node for an operation with an optional value."""

    _syntax_ = [
        "pynative.constant {arg.elements_attribute}",
        "pynative.constant {arg.float_attribute}",
        "pynative.constant {arg.integer_attribute}",
    ]


##############################################################################
# Dialect

DIALECT_PYNATIVE = Dialect(
    "pynative",
    ops=[PYNATIVE_ConstantOp],
    types=[],
    preamble="",
    transformers=None,
)
