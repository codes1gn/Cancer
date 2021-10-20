
#ifndef CANCER_BACKEND_IREE_PASSES_H
#define CANCER_BACKEND_IREE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace CANCER {
namespace IREEBackend {
/// Registers all IREEBackend passes.
void registerIREEBackendPasses();

std::unique_ptr<OperationPass<ModuleOp>> createLowerLinkagePass();

} // namespace IREEBackend
} // namespace CANCER
} // namespace mlir

#endif // CANCER_BACKEND_IREE_PASSES_H
