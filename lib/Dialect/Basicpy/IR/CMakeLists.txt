add_mlir_dialect_library(CANCERDialectBasicpy
  BasicpyDialect.cpp
  BasicpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Basicpy

  DEPENDS
  MLIRBasicpyOpsIncGen

  LINK_LIBS PUBLIC
  CANCERTypingCPA
  MLIRIR
)
