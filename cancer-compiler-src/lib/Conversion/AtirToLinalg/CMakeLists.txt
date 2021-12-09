add_cancer_conversion_library(CANCERAtirToLinalg
  AtirToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cancer/Conversion/AtirToLinalg

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  MLIRMemRef
  CANCERDialectAtir
)
