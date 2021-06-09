add_subdirectory(Runtime)
add_subdirectory(JITHelpers)

add_cancer_library(CANCERRefBackend
  RefBackend.cpp
  LowerToLLVM.cpp
  LowerToRefbackrtABI.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/cancer/RefBackend

  DEPENDS
  CANCERRefBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRSCFToStandard
  MLIRSCFTransforms
  MLIRShapeToStandard
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  )

mlir_check_all_link_libraries(CANCERRefBackend)
