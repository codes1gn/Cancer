""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from cancer_frontend.python import PythonRunner
from typing import Callable, List
Vector = List[float]


def analyse(the_func: Callable) -> None:

    # TODO wrapper this logics into functions
    pyast = PythonRunner.parse_python(the_func)
    print(PythonRunner.dump_python(pyast))
    mlast = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(mlast))


def test_constant():

    # define test function
    def add_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 + var0
        return res
    
    def add_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 + arg1
        return res

    def add_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 + var1
        return res
    
    def listAdd0(arg0: list, arg1: list) -> list:
        res = arg0 + arg1
        return res
    
    def listAdd1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = arg0 + arg1
        return res
    
    # analyse(add_scalar0)
    # analyse(add_scalar1)
    # analyse(add_scalar2)
    # analyse(listAdd0)
    analyse(listAdd1)
if __name__ == "__main__":
    test_constant()