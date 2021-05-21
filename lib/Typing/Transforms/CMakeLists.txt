add_cancer_conversion_library(CANCERTypingPasses
  Passes.cpp
  CPATypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Typing/Transforms

  DEPENDS
  CANCERTypingTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CANCERTypingCPASupport
  CANCERTypingCPA
)
