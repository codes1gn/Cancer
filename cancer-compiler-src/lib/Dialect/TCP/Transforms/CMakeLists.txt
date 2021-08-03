add_cancer_conversion_library(CANCERTCPPasses
  Passes.cpp
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/TCP/Transforms

  DEPENDS
  CANCERTCPPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERDialectTCP
)
