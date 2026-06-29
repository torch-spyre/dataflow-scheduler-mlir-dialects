// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @receive_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      %1 = dataflow.get_unit {name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:      dataflow.program_unit iter_arg : %arg0 -> (%1) : {
// CHECK-NEXT:        %2 = dataflow.receive %0 : vector<64xf16>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @receive_test() {
    %src = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    %dst = dataflow.get_unit {name = "C0-SFU", type = "SFU"} : index
    dataflow.program_unit iter_arg : %arg -> (%dst) : {
      %data = dataflow.receive %src : vector<64xf16>
    }
    return
  }
}
