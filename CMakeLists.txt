cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# set backend option, use refjit
option(BUILD_CANCER_REFJIT "Enables the reference JIT backend." ON)
option(BUILD_CANCER_IREE_BACKEND "Enables the reference JIT backend." ON)
option(CANCER_BUILD_CANCER_DYLIB "Enables shared build of NPCOMP dylib (depends on LLVM/MLIR dylib support)" ON)
# TODO let this IREE builddir point to the build caches of IREE project
set(CANCER_IREE_BUILDDIR "./iree_build" CACHE STRING "read from iree build cache")

# build project as standalone structure, use mlir and llvm as external lib
project(cancer LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

# config mlir and llvm
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LIT_ARGS_DEFAULT "-sv")
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set mlir and llvm to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(EnvFuncs)
include(AddCancer)
include(CancerPython)

set(CANCER_BUILT_STANDALONE 1)
set(CANCER_TABLEGEN_ARGS "")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/cancer-compiler-src/include)
include_directories(${PROJECT_BINARY_DIR}/cancer-compiler-src/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# build IREE

if(BUILD_CANCER_IREE_BACKEND)
  add_compile_definitions(BUILD_CANCER_IREE_BACKEND)
  string(APPEND CANCER_TABLEGEN_ARGS "-DBUILD_CANCER_IREE_BACKEND")
  if (CANCER_IREE_BUILDDIR)
    message(STATUS "importing IREE from iree_build")
    set(IREE_DIR "${CANCER_IREE_BUILDDIR}")
    set(IREE_FOUND 1)
  else()
    message(FATAL_ERROR "currently not tuned installed package of IREE")
  endif()

  # before IREE install solution done, use this way to debug with iree-build instead
  function(link_iree_objs src dst)
    set(src_path "${IREE_DIR}/${src}")
    set(dst_path "${CMAKE_BINARY_DIR}/${dst}")
    get_filename_component(src_path "${src_path}" ABSOLUTE)
    get_filename_component(dst_dir "${dst_path}" PATH)
    file(MAKE_DIRECTORY "${dst_dir}")
    execute_process(
      COMMAND
      ${CMAKE_COMMAND} -E create_symlink "${src_path}" "${dst_path}"
      RESULT_VARIABLE result
    )
    if(NOT ${result} EQUAL 0)
      message(FATAL_ERROR "FAIL: cannot link iree build cache")
    endif()
  endfunction()
  link_iree_objs(iree/tools/iree-translate backend_iree/tools/iree-translate)
  link_iree_objs(bindings/python/pyiree/rt backend_iree/pyiree/rt)
endif()

# look for python env
set(CANCER_PYTHON_BINDINGS_VERSION_LOCKED 1 CACHE BOOL
    ¦  "Links to specific python libraries, resolving all symbols.")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Found python libraries: ${Python_LIBRARIES}")



# build subcomponents
add_subdirectory(external/pybind11)
add_subdirectory(test)
add_subdirectory(cancer-compiler-src)

# add cancer-compiler utils
add_subdirectory(cancer-compiler)

