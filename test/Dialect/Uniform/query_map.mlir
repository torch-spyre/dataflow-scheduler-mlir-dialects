// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// uniform.query_map consumes a uniform.def_immutable_mapping result.
// Both ops are built here so the mapping is available for the query.

// CHECK-LABEL:   func.func @query_map_test() {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[MAP:.*]] = uniform.def_immutable_mapping([%[[C0]] -> %[[C0]]], [%[[C1]] -> %[[C1]]]):index
// CHECK-NEXT:      %[[Q:.*]] = uniform.query_map(map:%[[MAP]], key:%[[C0]]) : index
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @query_map_test() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %c0], [%c1 -> %c1]):index
    %q = uniform.query_map(map:%map, key:%c0) : index
    return
  }
}
