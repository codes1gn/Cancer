add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(RefBackend)
add_subdirectory(Typing)
add_subdirectory(capi)



# TODO fix this init process to opt not to init
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(cancer_dialect_libs GLOBAL PROPERTY CANCER_DIALECT_LIBS)
get_property(cancer_conversion_libs GLOBAL PROPERTY CANCER_CONVERSION_LIBS)

message(STATUS "CANCER Dialect libs: ${cancer_dialect_libs}")
message(STATUS "CANCER Conversion libs: ${cancer_conversion_libs}")

add_cancer_library(CANCERInit
    CancerInit.cpp

  LINK_LIBS

  PUBLIC
  # Local depends
  CANCERDialectBasicpy
  CANCERDialectNumpy
  CANCERDialectTCF
  CANCERDialectTCP
  CANCERDialectRefback
  CANCERDialectRefbackrt

  CANCERBasicpyPasses
  CANCERNumpyPasses
  CANCERTCFPasses
  CANCERTCPPasses

  CANCERConversionPasses
  CANCERRefBackend

  ${cancer_dialect_libs}
  ${cancer_conversion_libs}
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
)

