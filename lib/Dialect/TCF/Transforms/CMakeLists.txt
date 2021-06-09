add_cancer_conversion_library(CANCERTCFPasses
  Passes.cpp
  ShapeRefinement.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/TCF/Transforms

  DEPENDS
  CANCERTCFPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERDialectTCF
)
