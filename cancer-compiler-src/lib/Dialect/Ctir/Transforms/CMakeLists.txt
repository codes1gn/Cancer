add_cancer_conversion_library(CANCERCtirPasses
  Passes.cpp
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Ctir/Transforms

  DEPENDS
  CANCERCtirPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERDialectCtir
)
