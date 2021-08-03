add_cancer_dialect_library(CANCERDialectTCF
  TCFDialect.cpp
  TCFOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/TCF

  DEPENDS
  MLIRTCFOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
