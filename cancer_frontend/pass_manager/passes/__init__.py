from . import *

import inspect
import sys

print(inspect.getmembers(sys.modules[__name__]))
# assert 0
# lambda obj: is_op(obj, __name__)
