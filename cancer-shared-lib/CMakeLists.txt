# Building libmlir-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  message(WARNING "Not building CANCER dylib because PIC is disabled")
  return()
endif()

# Building libmlir-cpp.so may not work on MSVC
if (MSVC)
  message(WARNING "Not building CANCER dylib because not yet supported on MSVC")
  return()
endif()

if(NOT CANCER_BUILD_CANCER_DYLIB)
  message(WARNING "Not building CANCER dylib (not CANCER_BUILD_CANCER_DYLIB): Fully static builds not yet supported")
  return()
endif()

get_property(cancer_libs GLOBAL PROPERTY CANCER_STATIC_LIBS)
list(REMOVE_DUPLICATES cancer_libs)

# Populate _OBJECTS and _DEPS as necessary per platform.
foreach (lib ${cancer_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # Add transitive deps explcitly since otherwise, there would just be
  # objects.
  list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach()

# Note: Does not use add_cancer_library, which is used for things that go
# *into* the libCANCER.so. This is building the shared library, so use
# a higher-level rule.
llvm_add_library(
  CANCERSHLIB
  SHARED
  cancer-shared-lib.cpp
  ${_OBJECTS}
  LINK_LIBS PUBLIC
  # Public dependencies on the MLIR public API and impl shared libraries.
  MLIRPublicAPI
  MLIR
  ${_DEPS}
)
target_link_libraries(CANCERSHLIB PRIVATE ${LLVM_PTHREAD_LIB})
