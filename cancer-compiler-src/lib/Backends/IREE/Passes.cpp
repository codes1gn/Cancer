
#include "PassDetail.h"
#include "Backends/IREE/Passes.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::CANCER;
using namespace mlir::CANCER::IREEBackend;

namespace {
// This pass lowers the public ABI of the module to the primitives exposed by
// the refbackrt dialect.
class LowerLinkagePass : public LowerLinkageBase<LowerLinkagePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto func : module.getOps<FuncOp>()) {
      if (func.getVisibility() == SymbolTable::Visibility::Public)
        func->setAttr("iree.module.export", UnitAttr::get(&getContext()));
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::CANCER::IREEBackend::createLowerLinkagePass() {
  return std::make_unique<LowerLinkagePass>();
}
