add_cancer_dialect_library(CANCERDialectCtir
  CtirDialect.cpp
  CtirOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Ctir

  DEPENDS
  MLIRCtirOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
