// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK:         #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @get_logical_memory_view_test() {
// CHECK-NEXT:      %0 = dataflow.get_unit {name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:      %c128 = arith.constant 128 : index
// CHECK-NEXT:      %1 = dataflow.get_logical_memory_view %0, %c128 {layout_map = #map} : index, index, memref<128xi8>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

module {
  func.func @get_logical_memory_view_test() {
    %u = dataflow.get_unit {name = "C0-MNILU", type = "MNILU"} : index
    %c128 = arith.constant 128 : index
    %view = dataflow.get_logical_memory_view %u, %c128 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<128xi8>
    return
  }
}
