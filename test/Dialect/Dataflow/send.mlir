// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @send_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      %1 = dataflow.get_unit {name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
// CHECK-NEXT:        %cst = arith.constant dense<0.000000e+00> : vector<64xf16>
// CHECK-NEXT:        dataflow.send %1, %cst : vector<64xf16>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @send_test() {
    %src = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    %dst = dataflow.get_unit {name = "C0-SFU", type = "SFU"} : index
    dataflow.program_unit iter_arg : %arg -> (%src) : {
      %val = arith.constant dense<0.0> : vector<64xf16>
      dataflow.send %dst, %val : vector<64xf16>
    }
    return
  }
}
