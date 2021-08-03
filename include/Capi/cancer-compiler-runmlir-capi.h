//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility binary for compiling and running code through the cancer
// compiler/runtime stack.
//
//===----------------------------------------------------------------------===//

#ifndef CANCER_PYRUNNER_CANCER_RUNNER_BACKEND_H
#define CANCER_PYRUNNER_CANCER_RUNNER_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

int cancerrun(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif // CANCER_PYRUNNER_CANCER_RUNNER_BACKEND_H
