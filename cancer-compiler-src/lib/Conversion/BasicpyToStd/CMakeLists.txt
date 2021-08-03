add_cancer_conversion_library(CANCERBasicpyToSTD
  Passes.cpp
  PrimitiveOpsConversion.cpp

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  CANCERDialectBasicpy
)
