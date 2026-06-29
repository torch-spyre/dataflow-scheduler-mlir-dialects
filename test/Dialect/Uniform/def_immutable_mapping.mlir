// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @def_immutable_mapping_test() {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[MAP:.*]] = uniform.def_immutable_mapping([%[[C0]] -> %[[C0]]], [%[[C1]] -> %[[C1]]]):index
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @def_immutable_mapping_test() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %c0], [%c1 -> %c1]):index
    return
  }
}
