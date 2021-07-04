# cancer_runner is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.

if(CANCER_PYTHON_BINDINGS_VERSION_LOCKED)
  set(CANCER_PYEXT_LINK_MODE SHARED)
  set(CANCER_PYEXT_LIBADD ${Python3_LIBRARIES})
else()
  set(CANCER_PYEXT_LINK_MODE MODULE)
  set(CANCER_PYEXT_LIBADD)
endif()

message(STATUS "CANCER_PYEXT_LINK_MODE = :${CANCER_PYEXT_LINK_MODE}")
message(STATUS "CANCER_PYEXT_LIBADD = :${CANCER_PYEXT_LIBADD}")

# TODO fix this init process to opt not to init
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(cancer_dialect_libs GLOBAL PROPERTY CANCER_DIALECT_LIBS)
get_property(cancer_conversion_libs GLOBAL PROPERTY CANCER_CONVERSION_LIBS)

message(STATUS "CANCER Dialect libs: ${cancer_dialect_libs}")
message(STATUS "CANCER Conversion libs: ${cancer_conversion_libs}")



# TODO(albert) use cancer_pyrunner as target name for resolve python 
# slash issue
pybind11_add_module(cancer_pyrunner cancer-pyrunner.cpp)
# add_library(cancer_pyrunner ${CANCER_PYEXT_LINK_MODE} 
#     cancer-pyrunner.cpp
# )

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(cancer_pyrunner PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

set_target_properties(cancer_pyrunner PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                      "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(cancer_pyrunner PROPERTIES OUTPUT_NAME cancer_pyrunner)
set_target_properties(cancer_pyrunner PROPERTIES PREFIX
                      "${PYTHON_MODULE_PREFIX}")
set_target_properties(cancer_pyrunner PROPERTIES SUFFIX
                      "${PYTHON_MODULE_EXTENSION}")

message(STATUS "CANCERSHLIB is ${CANCERSHLIB}")
# pybind requires binding code to be compiled with -fvisibility=hidden
# Better code can be generated if the entire project compiles that way, but
# that is not enforced here.
set_target_properties(cancer_pyrunner PROPERTIES CXX_VISIBILITY_PRESET "hidden")

target_link_libraries(cancer_pyrunner
  PRIVATE
    # Transitive dep on the shared library links most things from there.
    # Shared library deps first ensure we get most of what we need from libraries.
    CANCERSHLIB

    cancer-runner-backend
    ${CANCER_PYEXT_LIBADD}
  )
cancer_python_target_compile_options(cancer_pyrunner)

mlir_check_all_link_libraries(cancer_pyrunner)
