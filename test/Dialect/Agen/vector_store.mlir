// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[SET:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL:   func.func @vector_store_basic(
// CHECK-SAME:      %[[ARG0:.*]]: vector<64xf16>, %[[ARG1:.*]]: memref<64xf16>) {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      agen.vector_store %[[ARG0]], %[[ARG1]][%[[C0]]] {store_order = #[[MAP]], store_set = #[[SET]]} : memref<64xf16>, vector<64xf16>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

module {
  func.func @vector_store_basic(%arg0: vector<64xf16>, %arg1: memref<64xf16>) {
    %c0 = arith.constant 0 : index
    agen.vector_store %arg0, %arg1[%c0] {store_order = #map, store_set = #set} : memref<64xf16>, vector<64xf16>
    return
  }
}
