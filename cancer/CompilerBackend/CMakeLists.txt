
pybind11_add_module(cancer_compiler cancer_compiler.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(cancer_compiler PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
