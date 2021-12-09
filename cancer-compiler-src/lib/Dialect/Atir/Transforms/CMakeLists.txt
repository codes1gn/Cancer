add_cancer_conversion_library(CANCERAtirPasses
  Passes.cpp
  ShapeRefinement.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Dialect/Atir/Transforms

  DEPENDS
  CANCERAtirPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERDialectAtir
)
