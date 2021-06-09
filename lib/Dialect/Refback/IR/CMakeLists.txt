add_cancer_dialect_library(CANCERDialectRefback
  RefbackDialect.cpp
  RefbackOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Refback

  DEPENDS
  MLIRRefbackOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
