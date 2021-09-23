from .pass_manager_base import PassManagerBase

from .passes.plugin_multiply_pass import PluginMultiplyPass

__all__ = [
    "PastToMlirPassManager",
]


class PastToMlirPassManager(PassManagerBase):
    def register_passes(self):
        print("PastToMlirPassManager::register_passes")
        self.add_pass(PluginMultiplyPass)

        return
