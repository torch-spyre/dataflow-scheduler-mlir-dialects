// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @create_group_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      %1 = dataflow.get_unit {name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      %2 = dataflow.create_group(%0, %1 : index, index) : index
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @create_group_test() {
    %u0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    %u1 = dataflow.get_unit {name = "C1-L1LU", type = "L1LU"} : index
    %g = dataflow.create_group (%u0, %u1 : index, index) : index
    return
  }
}
