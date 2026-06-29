// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// Negative: only 1 candidate.

func.func @select_memref_one_candidate() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, "L1">
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    // expected-error@+1 {{requires at least 2 candidate memrefs}}
    %sel = ktdf.select_memref %phase[%x0] : memref<64xf16, "L1">
  }
  memref.dealloc %x0 : memref<64xf16, "L1">
  return
}

// -----

// Negative: candidates with different types.

func.func @select_memref_mismatched_types() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, "L1">
  // expected-note@+1 {{prior use here}}
  %x1 = memref.alloc() : memref<32xf16, "L1">
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    // expected-error@+1 {{expects different type}}
    %sel = ktdf.select_memref %phase[%x0, %x1] : memref<64xf16, "L1">
  }
  memref.dealloc %x0 : memref<64xf16, "L1">
  memref.dealloc %x1 : memref<32xf16, "L1">
  return
}

// -----

// Negative: num_phases mismatches candidate count.

func.func @select_memref_num_phases_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, "L1">
  %x1 = memref.alloc() : memref<64xf16, "L1">
  %x2 = memref.alloc() : memref<64xf16, "L1">
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    // expected-error@+1 {{phase op num_phases (2) must equal candidate count (3)}}
    %sel = ktdf.select_memref %phase[%x0, %x1, %x2] : memref<64xf16, "L1">
  }
  memref.dealloc %x0 : memref<64xf16, "L1">
  memref.dealloc %x1 : memref<64xf16, "L1">
  memref.dealloc %x2 : memref<64xf16, "L1">
  return
}
