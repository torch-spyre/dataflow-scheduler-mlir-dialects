// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @binary_add(
// CHECK-SAME:      %arg0: vector<64xf16>, %arg1: vector<64xf16>) -> vector<64xf16> {
// CHECK-NEXT:      %0 = vectorchain.binary %arg0, %arg1 {binary_op = #vectorchain<binary_operator add>, op_specific_map = #map} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:      return %0 : vector<64xf16>
// CHECK-NEXT:    }

module {
  func.func @binary_add(%op1: vector<64xf16>, %op2: vector<64xf16>) -> vector<64xf16> {
    %result = vectorchain.binary %op1, %op2 {binary_op = #vectorchain<binary_operator add>, op_specific_map = affine_map<(d0) -> (d0)>} : vector<64xf16>, vector<64xf16>, vector<64xf16>
    return %result : vector<64xf16>
  }
}
