add_cancer_conversion_library(CANCERAtirToCtir
  AtirToCtir.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Conversion/AtirToCtir

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  CANCERDialectAtir
  CANCERDialectCtir
)
