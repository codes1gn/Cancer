add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToAtir)
add_subdirectory(AtirToLinalg)
add_subdirectory(AtirToStd)
add_subdirectory(AtirToCtir)
add_subdirectory(AtirToTosa)

add_cancer_library(CANCERConversionPasses
  Passes.cpp

  DEPENDS
  CANCERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CANCERBasicpyToSTD
  CANCERNumpyToAtir
  CANCERAtirToCtir
  CANCERAtirToTosa
)
