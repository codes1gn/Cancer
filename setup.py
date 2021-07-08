# -*- coding: utf-8 -*-
import os
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # CMake lets you override the generator - we need to check this.
        cmake_generator = "-GNinja"

        build_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not build_dir.endswith(os.path.sep):
            build_dir += os.path.sep

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            # TODO(albert) make it none hardcode
            "-DMLIR_DIR={}".format("/root/coding/Cancer/mlir_build/install_dir/lib/cmake/mlir"),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DLLVM_EXTERNAL_LIT={}".format("/root/coding/Cancer/mlir_build/bin/llvm-lit"),
            "-DCMAKE_BUILD_TYPE={}".format("DEBUG"),  # not used on MSVC, but no harm
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(build_dir),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
        ]
        build_args = []
        build_args += ["-j8"]
        build_args += ["--verbose"]
        # build_args += ["--clean-first"]
        build_args += ["--target", "cancer_pyrunner"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)

        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        subprocess.check_call(
            ["cp", "./cancer-pyrunner/cancer_pyrunner.cpython-36m-x86_64-linux-gnu.so", build_dir], cwd=self.build_temp
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="cancer",
    version="0.0.1",
    author="Albert Shi, Tianyu Jiang",
    author_email="codefisheng@gmail.com",
    description="Composite AI Compiler Experiment Platform",
    long_description="",
    ext_modules=[CMakeExtension("cancer_pyrunner")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.6.12",
    entry_points={
        "console_scripts": [
            "cancer_runner=cancer.bin.cancer_runner:main",
        ]
    },
)
