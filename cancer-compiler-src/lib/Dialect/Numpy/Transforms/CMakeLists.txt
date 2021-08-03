add_cancer_conversion_library(CANCERNumpyPasses
  Passes.cpp
  PublicFunctionToTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Numpy/Transforms

  DEPENDS
  CANCERNumpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERDialectNumpy
)
