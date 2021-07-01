# cancer-runner is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.


get_property(dialect_libs GLOBAL PROPERTY CANCER_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY CANCER_CONVERSION_LIBS)

add_cancer_executable(cancer-pyrunner
  cancer-pyrunner.cpp
  )

llvm_update_compile_flags(cancer-pyrunner)
target_link_libraries(cancer-pyrunner PRIVATE
  # Shared library deps first ensure we get most of what we need from libraries.
  CANCERSHLIB
  MLIR

  CANCERCAPI
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRJitRunner
  MLIRParser
  MLIRSupport
  CANCERInit
  CANCERRefBackendJITHelpers
  ${conversion_libs}
  ${dialect_libs}
)
add_dependencies(cancer-pyrunner
  CANCERCompilerRuntimeShlib
  CANCERSHLIB
  )
