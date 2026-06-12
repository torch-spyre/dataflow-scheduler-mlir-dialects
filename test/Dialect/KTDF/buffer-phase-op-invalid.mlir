// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// Negative: num_phases < 2.

func.func @buffer_phase_num_phases_too_small() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // expected-error@+1 {{minimum value is 2}}
    %phase = ktdf.buffer_phase(%i) {num_phases = 1 : i64} : index
  }
  return
}

// -----

// Negative: empty operand list.

func.func @buffer_phase_no_ivs() {
  // expected-error@+1 {{requires at least one induction variable operand}}
  %phase = ktdf.buffer_phase() {num_phases = 2 : i64} : index
  return
}

// -----

// Negative: operand is not an scf.for IV.

func.func @buffer_phase_not_an_iv() {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{operand #0 is not an scf.for induction variable}}
  %phase = ktdf.buffer_phase(%c0) {num_phases = 2 : i64} : index
  return
}
