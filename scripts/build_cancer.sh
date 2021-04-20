cmake -G Ninja .. -DMLIR_DIR=/root/mlir_tutorials/llvm-project/build/install_dir/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/root/mlir_tutorials/llvm-project/build/bin/llvm-lit

cmake --build . --target standalone-opt
