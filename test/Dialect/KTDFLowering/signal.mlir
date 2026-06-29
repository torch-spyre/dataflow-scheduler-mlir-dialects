// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @signal_basic() {
// CHECK-NEXT:     ktdf_lowering.signal {applicable_units = ["MNILU", "L1LU"]}
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func @signal_basic() {
    ktdf_lowering.signal {applicable_units = ["MNILU", "L1LU"]}
    return
  }
}
