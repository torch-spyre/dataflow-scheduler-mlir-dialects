// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @get_unit_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @get_unit_test() {
    %u = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    return
  }
}
