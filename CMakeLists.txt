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
option(CANCER_BUILD_CANCER_DYLIB "Enables shared build of NPCOMP dylib (depends on LLVM/MLIR dylib support)" ON)

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
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# look for python env
set(CANCER_PYTHON_BINDINGS_VERSION_LOCKED 1 CACHE BOOL
    ¦  "Links to specific python libraries, resolving all symbols.")
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Found python libraries: ${Python_LIBRARIES}")

# look for pybind11 env
cancer_detect_pybind11_install()
find_package(pybind11 2.6 CONFIG REQUIRED)
set(pybind11_INCLUDE_DIR ${pybind11_INCLUDE_DIR})
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
    ¦   ¦   ¦  "suffix = '${PYTHON_MODULE_SUFFIX}', "
    ¦   ¦   ¦  "extension = '${PYTHON_MODULE_EXTENSION}")


# build subcomponents
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(cancer-opt)
add_subdirectory(cancer-translate)
add_subdirectory(cancer-shared-lib)
add_subdirectory(cancer-runner)
# include in order, to ensure targets of pybind11 are included
add_subdirectory(external/pybind11)
add_subdirectory(cancer)

