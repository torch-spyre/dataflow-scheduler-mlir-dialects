// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[SET:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL:   func.func @vector_load_basic(
// CHECK-SAME:      %[[ARG0:.*]]: memref<64xf16>) -> vector<64xf16> {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[RESULT:.*]] = agen.vector_load %[[ARG0]][%[[C0]]] {load_order = #[[MAP]], load_set = #[[SET]]} : memref<64xf16>, vector<64xf16>
// CHECK-NEXT:      return %[[RESULT]] : vector<64xf16>
// CHECK-NEXT:    }

#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

module {
  func.func @vector_load_basic(%arg0: memref<64xf16>) -> vector<64xf16> {
    %c0 = arith.constant 0 : index
    %0 = agen.vector_load %arg0[%c0] {load_order = #map, load_set = #set} : memref<64xf16>, vector<64xf16>
    return %0 : vector<64xf16>
  }
}
