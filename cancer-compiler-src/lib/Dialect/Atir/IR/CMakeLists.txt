add_cancer_dialect_library(CANCERDialectAtir
  AtirDialect.cpp
  AtirOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Atir

  DEPENDS
  MLIRAtirOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
