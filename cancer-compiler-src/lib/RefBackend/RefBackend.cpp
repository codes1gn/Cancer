//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base file for cancer's "reference backend".
//
// The input to this backend is a layer that we call "Ctir" + a mix of scalar
// ops. Ctir is currently a concrete dialect, but more generally it refers to a
// layer of the compilation stack consisting of named ops on entire tensors,
// with their preconditions checked. For example, a "matmul" op that assumes
// that the contracting ("k") dimensions of both operands are equal. Earlier
// code in the compilation stack should ensure that these preconditions are met
// (such as during Atir->Ctir lowering).
//
// The output of this backend is LLVM IR suitable for JITing.
//
// We expect that other backends will appear that have a similar kind of
// interface (Ctir + scalar ops ---> LLVM IR / other "executable").
//
//===----------------------------------------------------------------------===//

#include "RefBackend/RefBackend.h"
#include "PassDetail.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
// to fix memref ops
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "Conversion/AtirToLinalg/AtirToLinalg.h"
#include "Conversion/AtirToStd/AtirToStd.h"
#include "Conversion/AtirToCtir/AtirToCtir.h"
#include "Dialect/Refback/IR/RefbackOps.h"
#include "Dialect/Ctir/IR/CtirDialect.h"
#include "Dialect/Ctir/IR/CtirOps.h"
#include "Dialect/Ctir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::CANCER;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "RefBackend/Passes.h.inc"
} // end namespace

void mlir::CANCER::registerRefBackendPasses() {
  ::registerPasses();

  mlir::PassPipelineRegistration<RefBackendLoweringPipelineOptions>(
      "refback-lowering-pipeline", "RefBackend lowering pipeline.",
      mlir::CANCER::createRefBackendLoweringPipeline);
  // TODO: Move this out of RefBackend once the Atir->Ctir conversions
  // become more substantial.
  mlir::PassPipelineRegistration<RefBackendLoweringPipelineOptions>(
      "refback-atir-to-ctir-pipeline",
      "RefBackend lowering pipeline converting Atir ops to Ctir-level ops (not "
      "just Ctir dialect).",
      mlir::CANCER::createRefBackendAtirToCtirPipeline);
  mlir::PassPipelineRegistration<RefBackendLoweringPipelineOptions>(
      "atir-refback-lowering-pipeline",
      "RefBackend lowering pipeline, starting from Atir. (equivalent to "
      "refback-atir-to-ctir-pipeline + refback-lowering-pipeline)",
      mlir::CANCER::createAtirRefBackendLoweringPipeline);
}

//===----------------------------------------------------------------------===//
// LowerAllocMemRefOps
//===----------------------------------------------------------------------===//

namespace {
class LowerAllocMemRefOp : public OpRewritePattern<refback::AllocMemRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(refback::AllocMemRefOp op,
                                PatternRewriter &rewriter) const override {
    auto memrefType = op.getType().cast<MemRefType>();
    auto shape = op.getOperand();
    // std.alloc only accepts the dynamic extents as operands, so only
    // collect those.
    SmallVector<Value, 6> dynamicExtents;
    for (int i = 0, e = memrefType.getRank(); i < e; i++) {
      if (memrefType.isDynamicDim(i)) {
        auto ci = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
        auto extent = rewriter.create<tensor::ExtractOp>(op.getLoc(), shape,
                                                         ValueRange({ci}));
        dynamicExtents.push_back(extent);
      }
    }
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType, dynamicExtents);
    return success();
  }
};
} // namespace

namespace {
class LowerAllocMemRefOps
    : public LowerAllocMemRefOpsBase<LowerAllocMemRefOps> {

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    // OwningRewritePatternList patterns;
    // changed into new interface
    RewritePatternSet patterns(context);
    patterns.insert<LowerAllocMemRefOp>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<refback::AllocMemRefOp>();
    target.addLegalOp<tensor::ExtractOp>();
    target.addLegalOp<memref::AllocOp>();
    target.addLegalOp<ConstantOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CANCER::createLowerAllocMemRefOpsPass() {
  return std::make_unique<LowerAllocMemRefOps>();
}

//===----------------------------------------------------------------------===//
// RestrictedCanonicalizer
//===----------------------------------------------------------------------===//

namespace {
struct RestrictedCanonicalizer
    : public RestrictedCanonicalizerBase<RestrictedCanonicalizer> {
  void runOnOperation() override {
    auto *context = &getContext();

    // Find the dialects from their names.
    DenseSet<StringRef> neededDialects;
    for (const std::string &dialectName : includedDialects)
      neededDialects.insert(dialectName);
    DenseSet<Dialect *> dialectsToCanonicalize;
    for (Dialect *dialect : context->getLoadedDialects()) {
      if (neededDialects.count(dialect->getNamespace())) {
        dialectsToCanonicalize.insert(dialect);
        // Erase the dialect so that we can report an error below for any
        // dialect names that are not loaded.
        neededDialects.erase(dialect->getNamespace());
      }
    }

    // Report a helpful error if a dialect is not found.
    auto missingDialects = llvm::to_vector<6>(neededDialects);
    if (!missingDialects.empty()) {
      llvm::sort(missingDialects);
      std::string buf;
      llvm::raw_string_ostream os(buf);
      llvm::interleaveComma(missingDialects, os);
      llvm::report_fatal_error("restricted-canonicalize: unknown dialects: " +
                               os.str());
    }

    // Collect all canonicalization patterns from ops in the included dialects.
    RewritePatternSet patterns(context);
    for (AbstractOperation *op : context->getRegisteredOperations())
      if (dialectsToCanonicalize.count(&op->dialect))
        op->getCanonicalizationPatterns(patterns, context);

    Operation *op = getOperation();
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::CANCER::createRestrictedCanonicalizerPass() {
  return std::make_unique<RestrictedCanonicalizer>();
}

//===----------------------------------------------------------------------===//
// createRefBackendLoweringPipeline
//===----------------------------------------------------------------------===//

void mlir::CANCER::createRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options) {

  // Convert all elementwise ops to linalg.
  //
  // Considering correctness, this lets us reuse the linalg bufferization, which
  // applies uniformly to all linalg structured ops.
  //
  // Also, converting to linalg herevopens up a lot of optimization
  // opportunities.
  pm.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());

  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createLinalgFusionOfTensorOpsPass());
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // Lower shape constraints before we enter tensor->memref conversion.
  // That is, we expand shape.cstr_* ops to eager error handling code.
  pm.addNestedPass<FuncOp>(createConvertShapeConstraintsPass());
  // Run shape canonicalizations. In particular, this erases shape.assuming,
  // now that we have converted shape constraints.
  // TODO: This is kind of ugly. Either we use pass options or a constructor
  // that takes C++ data structures. The former makes the pass usable on the
  // command line (including reproducers), the latter makes the pass more
  // convenient.
  std::unique_ptr<Pass> shapeCanonicalizer =
      createRestrictedCanonicalizerPass();
  if (failed(shapeCanonicalizer->initializeOptions("included-dialects=shape")))
    llvm::report_fatal_error("couldn't initialize restricted-canonicalize");
  pm.addPass(std::move(shapeCanonicalizer));

  // Lower shape ops to std.
  pm.addPass(createConvertShapeToStandardPass());

  // --------------------------------------------------------------------------
  // Lower the `tensor` type to `memref`.
  // --------------------------------------------------------------------------
  // We make a conscious effort here to do this as a sequence of separate passes
  // rather than a single mega dialect conversion pass.
  //
  // This means that intermediate steps have source/target materializations
  // (tensor_load / tensor_to_memref) in the IR.

  // Run tensor constant bufferization.
  // This pass has to run on a module op, and so does the final
  // FuncBufferizePass. But everything else can run in parallel on functions,
  // so we try to bracket the entire bufferization pipeline with the module
  // passes to allow maximum parallelism.
  pm.addPass(createTensorConstantBufferizePass());
  pm.addNestedPass<FuncOp>(createCtirBufferizePass());
  // refback::AllocMemRefOp takes a shape (i.e. extent tensor) as an argument.
  // We need to resolve this to std.alloc which takes individual extents.
  pm.addNestedPass<FuncOp>(createLowerAllocMemRefOpsPass());
  pm.addNestedPass<FuncOp>(createSCFBufferizePass());
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<FuncOp>(createStdBufferizePass());
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());
  pm.addPass(createFuncBufferizePass());
  pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());

  // TODO: Do buffer deallocation. We should be able to just drop in the
  // upstream pass?

  // At this point, we have lots of loose stuff floating around from lowering,
  // so it's a good time to do some general cleanups.
  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Preparation for converting to an LLVM module.
  // --------------------------------------------------------------------------
  // Now, we begin the process of lowering to LLVM's level of abstraction
  // (after which LLVM will take over lowering to machine code).

  // Lower linalg ops to loops.
  // TODO: Do some linalg optimizations like tiling here.
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());

  // Run a some cleanups.
  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Final conversion to an LLVM module.
  // --------------------------------------------------------------------------

  // Convert affine to std control flow in preparation for going to LLVM.
  pm.addNestedPass<FuncOp>(createLowerAffinePass());

  // Convert scf to std control flow in preparation for going to LLVM.
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());

  // Convert functions signatures and other constructs that interface with the
  // runtime to the `refbackrt` dialect.
  pm.addPass(createLowerToRefbackrtABIPass());

  // Finally, convert to LLVM dialect using our custom LowerToLLVM pass
  // which reuses the upstream patterns and gives us a place to add our own
  // patterns for our own custom ops like the refbackrt ops.
  pm.addPass(createLowerToLLVMPass());

  // Although LLVM will clean everything up eventually, for the sake of IR
  // clarity while still in MLIR, run some cleanups.
  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }
}

void mlir::CANCER::createRefBackendAtirToCtirPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options) {
  // Convert from Atir dialect to Ctir-level ops.
  //
  // Atir has implicit broadcasting, and issues errors "inside the ops" in the
  // case of invalid broadcasts.
  //
  // Ctir-level ops do not. So we need to reify the broadcasting and error
  // checking.
  //
  // Note that Ctir-level ops includes ops outside the Ctir dialect itself, such
  // as std elementwise ops on tensors and linalg ops on tensors.
  pm.addNestedPass<FuncOp>(createConvertAtirToStdPass());
  pm.addNestedPass<FuncOp>(createConvertAtirToLinalgPass());
  pm.addNestedPass<FuncOp>(createConvertAtirToCtirPass());

  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }
}

void mlir::CANCER::createAtirRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options) {
  createRefBackendAtirToCtirPipeline(pm, options);
  createRefBackendLoweringPipeline(pm, options);
}
