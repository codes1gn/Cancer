add_cancer_conversion_library(CANCERBasicpyPasses
  Passes.cpp
  TypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Basicpy/Transforms

  DEPENDS
  CANCERBasicpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERTypingCPA
)
