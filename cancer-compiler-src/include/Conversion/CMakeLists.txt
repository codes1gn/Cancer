set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls ${CANCER_TABLEGEN_ARGS})
add_public_tablegen_target(CANCERConversionPassIncGen)

add_mlir_doc(Passes -gen-pass-doc CANCERConversionPasses ./)
