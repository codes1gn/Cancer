add_cancer_library(CANCERRefBackendJITHelpers
  JITModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/cancer/RefBackend/JITHelpers

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CANCERRuntime
  CANCERRefBackend
  MLIRExecutionEngine
  )

mlir_check_all_link_libraries(CANCERRefBackend)
