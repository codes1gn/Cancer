set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_cancer_library(CANCERCAPI
  InitLLVM.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRTargetLLVMIRExport
  CANCERInit
  CANCERDialectBasicpy
  CANCERDialectNumpy
  )
