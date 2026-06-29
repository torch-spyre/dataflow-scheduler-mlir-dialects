// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// dataflow.return is the implicit terminator of dataflow.program_unit regions.
// It is elided in the printer when it has no attributes, operands, or results.
// We exercise it via the generic form; the round-trip output shows an empty body.

// CHECK-LABEL:   func.func @return_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @return_test() {
    %u = dataflow.get_unit {name = "C0-L1LU", type = "L1LU"} : index
    dataflow.program_unit iter_arg : %arg -> (%u) : {
      "dataflow.return"() : () -> ()
    }
    return
  }
}
