// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[MAP1:.+]] = affine_map<() -> (0)>
// CHECK: #[[MAP2:.+]] = affine_map<() -> ()>
// CHECK: #[[SET:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[SET1:.+]] = affine_set<() : (0 == 0)>

// CHECK-LABEL:   func.func @composite_load_and_store_basic(
// CHECK-SAME:      %[[ARG0:.*]]: memref<64xf16>, %[[ARG1:.*]]: memref<64xf16>) {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      agen.composite_load_and_store src:%[[ARG0]][%[[C0]]] dst:%[[ARG1]][%[[C0]]]
// CHECK-NEXT:       time_symbols(), load_iv(%[[IV:.*]]:vector<64xf16>)
// CHECK-NEXT:       {load_order = #[[MAP]], load_set = #[[SET]], load_time_addr_map = #[[MAP1]], store_order = #[[MAP]], store_set = #[[SET]], store_time_addr_map = #[[MAP1]], time_order = #[[MAP2]], time_set = #[[SET1]]}
// CHECK-NEXT:      {
// CHECK-NEXT:        agen.yield
// CHECK-NEXT:      } : memref<64xf16>, memref<64xf16>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> ()>
#set = affine_set<(d0) : (d0 == 0)>
// Note: affine_set<() : ()> normalizes to affine_set<() : (0 == 0)> when printed.
#set1 = affine_set<() : ()>

module {
  func.func @composite_load_and_store_basic(%arg0: memref<64xf16>, %arg1: memref<64xf16>) {
    %c0 = arith.constant 0 : index
    agen.composite_load_and_store src:%arg0[%c0] dst:%arg1[%c0]
     time_symbols(), load_iv(%iv:vector<64xf16>)
     {load_order = #map, load_set = #set, load_time_addr_map = #map1, store_order = #map, store_set = #set, store_time_addr_map = #map1, time_order = #map2, time_set = #set1}
    {
      agen.yield
    } : memref<64xf16>, memref<64xf16>
    return
  }
}
