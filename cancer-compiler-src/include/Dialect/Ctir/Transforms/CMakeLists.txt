set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CANCERCtirPassIncGen)

add_mlir_doc(Passes CANCERCtirTransforms ./ -gen-pass-doc)

