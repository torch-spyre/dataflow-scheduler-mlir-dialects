// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @shift(
// CHECK-SAME:      %[[X:.*]]: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>, vector<32xi32>, vector<32xi32>) {
// CHECK-NEXT:      %[[L:.*]] = vectorchain.shift %[[X]] {amount = -3 : i32} : vector<32xi32>, vector<32xi32>
// CHECK-NEXT:      %[[R:.*]] = vectorchain.shift %[[X]] {amount = 3 : i32} : vector<32xi32>, vector<32xi32>
// CHECK-NEXT:      %[[L31:.*]] = vectorchain.shift %[[X]] {amount = -31 : i32} : vector<32xi32>, vector<32xi32>
// CHECK-NEXT:      %[[R31:.*]] = vectorchain.shift %[[X]] {amount = 31 : i32} : vector<32xi32>, vector<32xi32>
// CHECK-NEXT:      return %[[L]], %[[R]], %[[L31]], %[[R31]]
// CHECK-NEXT:    }

module {
  func.func @shift(%x: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>, vector<32xi32>, vector<32xi32>) {
    %left = vectorchain.shift %x {amount = -3 : i32} : vector<32xi32>, vector<32xi32>
    %right = vectorchain.shift %x {amount = 3 : i32} : vector<32xi32>, vector<32xi32>
    %left_max = vectorchain.shift %x {amount = -31 : i32} : vector<32xi32>, vector<32xi32>
    %right_max = vectorchain.shift %x {amount = 31 : i32} : vector<32xi32>, vector<32xi32>
    return %left, %right, %left_max, %right_max : vector<32xi32>, vector<32xi32>, vector<32xi32>, vector<32xi32>
  }
}
