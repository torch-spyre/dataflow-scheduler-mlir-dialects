// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// uniform.yield is a Terminator op with optional variadic operands.
// No op in the Uniform dialect has a body region that requires it, so we
// exercise it as the sole terminator of a func.func body (which accepts any
// Terminator-trait op as its block terminator).  The printer emits it as the
// bare token "uniform.yield" when there are no operands — identical to how
// agen.yield is exercised in test/Dialect/Agen/yield.mlir.

// CHECK-LABEL:   func.func @yield_test() {
// CHECK-NEXT:      uniform.yield
// CHECK-NEXT:    }

module {
  func.func @yield_test() {
    uniform.yield
  }
}
