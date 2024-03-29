# cancer_runner is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.

# TODO fix this init process to opt not to init
get_property(cancer_dialect_libs GLOBAL PROPERTY CANCER_DIALECT_LIBS)
get_property(cancer_conversion_libs GLOBAL PROPERTY CANCER_CONVERSION_LIBS)

message(STATUS "CANCER Dialect libs: ${cancer_dialect_libs}")
message(STATUS "CANCER Conversion libs: ${cancer_conversion_libs}")

# build cancer-runner-backend as the public lib
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_cancer_library(cancer-compiler-runmlir-capi
  cancer-compiler-runmlir.cpp

  LINK_LIBS PUBLIC
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

add_dependencies(cancer-compiler-runmlir-capi
  CANCERCompilerRuntimeShlib
  CANCERSHLIB
  )

add_cancer_executable(cancer-compiler-runmlir
  cancer-compiler-runmlir.cpp
  )

llvm_update_compile_flags(cancer-compiler-runmlir)
target_link_libraries(cancer-compiler-runmlir PRIVATE
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
add_dependencies(cancer-compiler-runmlir
  CANCERCompilerRuntimeShlib
  CANCERSHLIB
  )
