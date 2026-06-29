// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @sync_send_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      %1 = dataflow.get_unit {name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
// CHECK-NEXT:        dataflow.sync_send %1 : index
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @sync_send_test() {
    %u0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    %u1 = dataflow.get_unit {name = "C1-SFU", type = "SFU"} : index
    dataflow.program_unit iter_arg : %arg -> (%u0) : {
      dataflow.sync_send %u1 : index
    }
    return
  }
}
