add_cancer_conversion_library(CANCERAtirToTosa
  AtirToTosa.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Conversion/AtirToTosa

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
)
