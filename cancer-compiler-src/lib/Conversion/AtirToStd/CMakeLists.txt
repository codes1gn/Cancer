add_cancer_conversion_library(CANCERAtirToStd
  AtirToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Conversion/AtirToStd

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  MLIRStandard
  MLIRLinalg
  CANCERDialectAtir
)
