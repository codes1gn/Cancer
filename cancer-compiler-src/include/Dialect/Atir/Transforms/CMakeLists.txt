set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CANCERAtirPassIncGen)

add_mlir_doc(Passes CANCERAtirTransforms ./ -gen-pass-doc)

