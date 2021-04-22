# Cancer Dialect Example

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a standalone `opt`-like tool to operate on that dialect.

## Building

For simplicity, invoke well-ready scripts is the easiest entry way.

Ensure that you have installed clang & clang++ as the compiler or gcc/g++.

use `scripts/build_deps.sh` to build the llvm + mlir dependencies.

use `scripts/build_cancer.sh` to build the cancer-translate and cancer-opt executable binary.

use `scripts/test_runner.sh` to test the usecases with filecheck

use `scripts/build_doc.sh` to build the documentation according to `.td` table declarations.

More detailly, this setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run

```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-cancer
```

To build the documentation from the TableGen description of the dialect operations, run

```sh
cmake --build . --target mlir-doc
```

**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.
More easily, use `pip install filecheck && ln -s ${which filecheck} /usr/bin/FileCheck` to given the executable path of filecheck to cmake.
