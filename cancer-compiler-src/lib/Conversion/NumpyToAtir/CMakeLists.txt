add_cancer_conversion_library(CANCERNumpyToAtir
  Passes.cpp

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  CANCERDialectBasicpy
  CANCERDialectNumpy
)
