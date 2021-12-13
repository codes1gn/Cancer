""" Tests pyMLIR on examples that use the Toy dialect. """
import os
import numpy as np
from cancer_frontend.python import PythonRunner
from typing import Callable, List

Vector = List[float]


def analyse(the_func: Callable) -> None:

    # TODO wrapper this logics into functions
    pyast = PythonRunner.parse_python(the_func)
    print(PythonRunner.dump_python(pyast))
    mlast = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(mlast))


def test_binary():

    # TODO: Add OP
    cunt = 0
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
        
        res = np.array(arg0) + np.array(arg1)
        return res
    
    def listAdd1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) + np.array(arg1)
        return res
    
    # analyse(add_scalar0)
    # cunt +=1
    # analyse(add_scalar1)
    # cunt +=1
    # analyse(add_scalar2)
    # cunt +=1
    # analyse(listAdd0)
    # cunt +=1
    # analyse(listAdd1)
    # cunt +=1
    # print(f'Add pass: {cunt}')
    
    # TODO: Sub OP
    cunt_sub = 0
    # def sub_scalar0(arg0:float) -> float:
    #     var0 = 2.0
    #     res = arg0 - var0
    #     return res

    # def sub_scalar1(arg0:float, arg1:float) -> float:
    #     res = arg0 - arg1
    #     return res

    # def sub_scalar2() -> float:
    #     var0 = 1.0
    #     var1 = 2.0
    #     res = var0 - var1
    #     return res
    
    def listSub0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) - np.array(arg1)
        return res
    
    def listSub1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) - np.array(arg1)
        return res
    
    # analyse(sub_scalar0)
    # cunt_sub +=1
    # analyse(sub_scalar1)
    # cunt_sub +=1
    # analyse(sub_scalar2)
    # cunt_sub +=1
    # analyse(listSub0)
    # cunt_sub +=1
    # analyse(listSub1)
    # cunt_sub +=1
    # print(f'cunt_sub={cunt_sub}')
    
    # TODO: Mul OP
    cunt_mul= 0
    def mul_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 * var0
        return res

    def mul_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 * arg1
        return res

    def mul_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 * var1
        return res
    
    def listMul0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) * np.array(arg1)
        return res
    
    def listMul1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) * np.array(arg1)
        # res = arg0 * arg1
        return res
    
    # analyse(mul_scalar0)
    # cunt_mul +=1
    # analyse(mul_scalar1)
    # cunt_mul +=1
    # analyse(mul_scalar2)
    # cunt_mul +=1
    # analyse(listMul0)
    # cunt_mul +=1
    # analyse(listMul1)
    # cunt_mul +=1
    # print(f'cunt_mul={cunt_mul}')
    
if __name__ == "__main__":
    test_binary()