// RUN: dataflow-scheduler-dialects-opt --canonicalize --split-input-file -allow-unregistered-dialect %s | FileCheck %s

// Partial fold: a constant operand is dropped, the IV operand is kept.
// CHECK-LABEL:   func.func @buffer_phase_drop_constant
// CHECK:           %[[P:.*]] = ktdf.buffer_phase(%[[IV:.*]]) {num_phases = 2 : i64} : index
func.func @buffer_phase_drop_constant() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%c0, %i) {num_phases = 2 : i64} : index
    "test.consume"(%phase) : (index) -> ()
  }
  return
}

// -----

// All-constant fold: the op becomes a constant 0 and is erased.
// CHECK-LABEL:   func.func @buffer_phase_all_constant
// CHECK-NOT:       ktdf.buffer_phase
// CHECK:           %[[Z:.*]] = arith.constant 0 : index
// CHECK:           "test.consume"(%[[Z]])
func.func @buffer_phase_all_constant() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %phase = ktdf.buffer_phase(%c0, %c3) {num_phases = 2 : i64} : index
  "test.consume"(%phase) : (index) -> ()
  return
}
