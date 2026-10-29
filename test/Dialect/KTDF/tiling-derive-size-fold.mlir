// RUN: dataflow-scheduler-dialects-opt --canonicalize --split-input-file -allow-unregistered-dialect %s | FileCheck %s

// iv==0, exact multiple: folds to min(2,2) = 2.
// CHECK-LABEL: func.func @ds_iv0_exact
// CHECK-NOT:     ktdf.tiling.derive_size
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         "test.consume"(%[[C2]])
func.func @ds_iv0_exact() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = ktdf.tiling.derive_size [%c0 : %c2], total_size = %c2 : index
  "test.consume"(%0) : (index) -> ()
  return
}

// -----

// iv==0, total < tile: folds to min(8,3) = 3.
// CHECK-LABEL: func.func @ds_iv0_short
// CHECK-NOT:     ktdf.tiling.derive_size
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         "test.consume"(%[[C3]])
func.func @ds_iv0_short() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %0 = ktdf.tiling.derive_size [%c0 : %c8], total_size = %c3 : index
  "test.consume"(%0) : (index) -> ()
  return
}

// -----

// iv==0, tile < total: folds to min(3,8) = 3 (the tile_size branch).
// CHECK-LABEL: func.func @ds_iv0_tile_smaller
// CHECK-NOT:     ktdf.tiling.derive_size
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         "test.consume"(%[[C3]])
func.func @ds_iv0_tile_smaller() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %0 = ktdf.tiling.derive_size [%c0 : %c3], total_size = %c8 : index
  "test.consume"(%0) : (index) -> ()
  return
}

// -----

// non-zero constant iv, total not a multiple of tile: NOT folded (round-trips).
// CHECK-LABEL: func.func @ds_iv_nonzero
// CHECK:         ktdf.tiling.derive_size
func.func @ds_iv_nonzero() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %0 = ktdf.tiling.derive_size [%c1 : %c2], total_size = %c7 : index
  "test.consume"(%0) : (index) -> ()
  return
}

// -----

// real (non-constant) IV, total not a multiple of tile: NOT folded
// (round-trips).
// CHECK-LABEL: func.func @ds_real_iv
// CHECK:         ktdf.tiling.derive_size
func.func @ds_real_iv() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %0 = ktdf.tiling.derive_size [%iv : %c2], total_size = %c7 : index
    "test.consume"(%0) : (index) -> ()
  }
  return
}

// -----

// multi-level (two pairs): NOT folded (round-trips).
// CHECK-LABEL: func.func @ds_multilevel
// CHECK:         ktdf.tiling.derive_size
func.func @ds_multilevel() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %0 = ktdf.tiling.derive_size [%c0 : %c2], [%c0 : %c3], total_size = %c6 : index
  "test.consume"(%0) : (index) -> ()
  return
}

// -----

// real IV, total a multiple of tile: every tile is full, folds to tile = 2.
// CHECK-LABEL: func.func @ds_real_iv_exact
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         scf.for
// CHECK-NOT:       ktdf.tiling.derive_size
// CHECK:           "test.consume"(%[[C2]])
func.func @ds_real_iv_exact() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %0 = ktdf.tiling.derive_size [%iv : %c2], total_size = %c8 : index
    "test.consume"(%0) : (index) -> ()
  }
  return
}

// -----

// real IV, tile size 1: every tile is full, folds to 1, and the resulting
// single-trip inner loop is removed.
// CHECK-LABEL: func.func @ds_real_iv_unit_tile
// CHECK:         scf.for %[[IV:.*]] =
// CHECK-NOT:       ktdf.tiling.derive_size
// CHECK-NOT:       scf.for
// CHECK:           "test.consume"(%[[IV]])
func.func @ds_real_iv_unit_tile() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %iv = %c0 to %c3 step %c1 {
    %0 = ktdf.tiling.derive_size [%iv : %c1], total_size = %c3 : index
    scf.for %j = %c0 to %0 step %c1 {
      %1 = arith.addi %iv, %j : index
      "test.consume"(%1) : (index) -> ()
    }
  }
  return
}

// -----

// multi-level with tile size 1 can be 0 or 1: NOT folded (round-trips).
// CHECK-LABEL: func.func @ds_multilevel_unit_tile
// CHECK:         ktdf.tiling.derive_size
func.func @ds_multilevel_unit_tile() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c23 = arith.constant 23 : index
  scf.for %i0 = %c0 to %c5 step %c1 {
    scf.for %i1 = %c0 to %c5 step %c1 {
      %0 = ktdf.tiling.derive_size [%i0 : %c5], [%i1 : %c1], total_size = %c23 : index
      "test.consume"(%0) : (index) -> ()
    }
  }
  return
}

// -----

// real IV, total 0: NOT folded to the tile size (round-trips).
// CHECK-LABEL: func.func @ds_real_iv_zero_total
// CHECK:         ktdf.tiling.derive_size
func.func @ds_real_iv_zero_total(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %iv = %c0 to %n step %c1 {
    %0 = ktdf.tiling.derive_size [%iv : %c2], total_size = %c0 : index
    "test.consume"(%0) : (index) -> ()
  }
  return
}
