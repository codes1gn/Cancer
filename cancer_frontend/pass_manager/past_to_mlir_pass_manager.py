from cancer_frontend.pass_manager.pass_manager_base import PassManagerBase
from cancer_frontend.pass_manager.passes import *

__all__ = [
    "PastToMlirPassManager",
]


class PastToMlirPassManager(PassManagerBase):
    def register_passes(self):
        print("PastToMlirPassManager::register_passes")
        self.add_pass(IdenticalPastPass)
        self.add_pass(StatementConversionPass)

        return
