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

// non-zero constant iv: NOT folded (round-trips).
// CHECK-LABEL: func.func @ds_iv_nonzero
// CHECK:         ktdf.tiling.derive_size
func.func @ds_iv_nonzero() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %0 = ktdf.tiling.derive_size [%c1 : %c2], total_size = %c8 : index
  "test.consume"(%0) : (index) -> ()
  return
}

// -----

// real (non-constant) IV: NOT folded (round-trips).
// CHECK-LABEL: func.func @ds_real_iv
// CHECK:         ktdf.tiling.derive_size
func.func @ds_real_iv() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %0 = ktdf.tiling.derive_size [%iv : %c2], total_size = %c8 : index
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
