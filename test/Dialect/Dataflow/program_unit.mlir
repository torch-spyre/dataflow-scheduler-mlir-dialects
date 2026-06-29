// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @program_unit_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @program_unit_test() {
    %u = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    dataflow.program_unit iter_arg : %arg -> (%u) : {
    }
    return
  }
}
