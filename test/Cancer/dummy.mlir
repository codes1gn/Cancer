// RUN: cancer-opt %s | cancer-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = cancer.foo %{{.*}} : i32
        %res = cancer.foo %0 : i32
        return
    }
}
