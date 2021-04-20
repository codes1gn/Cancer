#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

cmake -G Ninja .. -DMLIR_DIR=${top_dir_realpath}/mlir_build/install_dir/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=${top_dir_realpath}/mlir_build/bin/llvm-lit

cmake --build . --target standalone-opt
