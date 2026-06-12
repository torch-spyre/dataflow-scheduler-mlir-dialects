// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Round-trip: select between 2 L1 memrefs.
// CHECK-LABEL:   func.func @select_memref_2_l1
// CHECK:           %[[X0:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK:           %[[X1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK:           scf.for %[[I:.*]] = {{.*}} {
// CHECK:             %[[PHASE:.*]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             %[[SEL:.*]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : memref<64xf16, "L1">

func.func @select_memref_2_l1() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, "L1">
  %x1 = memref.alloc() : memref<64xf16, "L1">
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1] : memref<64xf16, "L1">
  }
  memref.dealloc %x0 : memref<64xf16, "L1">
  memref.dealloc %x1 : memref<64xf16, "L1">
  return
}

// Round-trip: select among 4 candidates with num_phases=4.
// CHECK-LABEL:   func.func @select_memref_4_l1
// CHECK:           %[[SEL:.*]] = ktdf.select_memref %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<64xf16, "L1">

func.func @select_memref_4_l1() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, "L1">
  %x1 = memref.alloc() : memref<64xf16, "L1">
  %x2 = memref.alloc() : memref<64xf16, "L1">
  %x3 = memref.alloc() : memref<64xf16, "L1">
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 4 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1, %x2, %x3] : memref<64xf16, "L1">
  }
  memref.dealloc %x0 : memref<64xf16, "L1">
  memref.dealloc %x1 : memref<64xf16, "L1">
  memref.dealloc %x2 : memref<64xf16, "L1">
  memref.dealloc %x3 : memref<64xf16, "L1">
  return
}
