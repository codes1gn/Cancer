add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToTCF)
add_subdirectory(TCFToLinalg)
add_subdirectory(TCFToStd)
add_subdirectory(TCFToTCP)

add_cancer_library(CANCERConversionPasses
  Passes.cpp

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CANCERBasicpyToSTD
  CANCERNumpyToTCF
  CANCERTCFToTCP
)
