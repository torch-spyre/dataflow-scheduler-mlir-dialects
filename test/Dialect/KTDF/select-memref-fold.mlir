// RUN: dataflow-scheduler-dialects-opt --canonicalize --split-input-file -allow-unregistered-dialect %s | FileCheck %s

// Constant phase 0 selects the first candidate.
// CHECK-LABEL:   func.func @select_phase0
// CHECK-NOT:       ktdf.select_memref
// CHECK:           "test.consume"(%[[A:.*]])
func.func @select_phase0(%a: memref<4xf16>, %b: memref<4xf16>) {
  %c0 = arith.constant 0 : index
  %sel = ktdf.select_memref %c0[%a, %b] : memref<4xf16>
  "test.consume"(%sel) : (memref<4xf16>) -> ()
  return
}

// -----

// Constant phase 1 selects the second candidate.
// CHECK-LABEL:   func.func @select_phase1
// CHECK-NOT:       ktdf.select_memref
// CHECK:           "test.consume"(%[[B:.*]])
func.func @select_phase1(%a: memref<4xf16>, %b: memref<4xf16>) {
  %c1 = arith.constant 1 : index
  %sel = ktdf.select_memref %c1[%a, %b] : memref<4xf16>
  "test.consume"(%sel) : (memref<4xf16>) -> ()
  return
}

// -----

// All-constant buffer_phase composes: phase folds to 0, select picks first.
// CHECK-LABEL:   func.func @buffer_phase_then_select
// CHECK-NOT:       ktdf.buffer_phase
// CHECK-NOT:       ktdf.select_memref
func.func @buffer_phase_then_select(%a: memref<4xf16>, %b: memref<4xf16>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %phase = ktdf.buffer_phase(%c0, %c3) {num_phases = 2 : i64} : index
  %sel = ktdf.select_memref %phase[%a, %b] : memref<4xf16>
  "test.consume"(%sel) : (memref<4xf16>) -> ()
  return
}
