#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
mkdir ${top_dir_realpath}/build
cd ${top_dir_realpath}/build

cmake -G Ninja .. -DMLIR_DIR=${top_dir_realpath}/mlir_build/install_dir/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=${top_dir_realpath}/mlir_build/bin/llvm-lit -DCMAKE_BUILD_TYPE=DEBUG

cmake --build . --target cancer-opt
cmake --build . --target cancer-translate
cmake --build . --target cancer-runner

# build mlir doc
cmake --build . --target mlir-doc

