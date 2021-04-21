get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRCancer
        )
add_llvm_executable(cancer-opt cancer-opt.cpp)

llvm_update_compile_flags(cancer-opt)
target_link_libraries(cancer-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(cancer-opt)
