# Cancer Introduction

The Composite AI Compiler Experiment Platform (Cancer) built with composite modularized frontend, midware and runtime backend. It builds with more flexible ways that allows you register new breed of frontend/backend implementations and compare with each other. It also relies on MLIR to provide fruitful manifolds and toolchains that allows you play with the IR design of the compiler part.

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a standalone `opt`-like tool to operate on that dialect.

This projects also refers to the idea and implementations of some similar works, including:

1. mlir-npcomp: https://github.com/google/mlir-npcomp
2. Jax: https://github.com/google/jax
3. Swift for Tensorflow: https://github.com/tensorflow/swift
4. MLIR.jl: https://github.com/vchuravy/MLIR.jl


## Build and run

It support the a simple `python-like` installation with setuptools. This will install the standalone python modules into your OS envs.

Also, you can choose more customized ways by using the scripts in ready. Ensure that you have installed clang & clang++ as the compiler or gcc/g++.

* use `scripts/build_python_pkg.sh` to build the python wheel distribution package.

* use `scripts/build_deps.sh` to build the llvm + mlir dependencies.

* use `scripts/build_cancer.sh` to build the cancer-translate and cancer-opt executable binary.

* use `scripts/test_runner.sh` to test the usecases with filecheck

* use `scripts/build_doc.sh` to build the documentation according to `.td` table declarations.

More detailly, this setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run

```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target <cancer-runner/cancer-opt/cancer-translate>
```

To build the documentation from the TableGen description of the dialect operations, run

```sh
cmake --build . --target mlir-doc
```

**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.
More easily, use `pip install filecheck && ln -s ${which filecheck} /usr/bin/FileCheck` to given the executable path of filecheck to cmake.
