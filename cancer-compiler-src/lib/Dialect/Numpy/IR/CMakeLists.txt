add_cancer_dialect_library(CANCERDialectNumpy
  NumpyDialect.cpp
  NumpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Numpy

  DEPENDS
  MLIRNumpyOpsIncGen

  LINK_LIBS PUBLIC
  CANCERDialectBasicpy
  MLIRIR
)
