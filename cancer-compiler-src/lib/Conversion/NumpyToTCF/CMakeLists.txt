add_cancer_conversion_library(CANCERNumpyToTCF
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
