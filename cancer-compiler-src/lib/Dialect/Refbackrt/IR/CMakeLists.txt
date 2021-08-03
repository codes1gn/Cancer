add_cancer_dialect_library(CANCERDialectRefbackrt
  RefbackrtDialect.cpp
  RefbackrtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Refbackrt

  DEPENDS
  MLIRRefbackrtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
