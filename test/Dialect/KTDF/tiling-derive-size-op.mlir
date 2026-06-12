// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Round-trip: 1-level, dynamic N and tile_size.
// CHECK-LABEL: func.func @tiling.derive_size_1level
func.func @tiling.derive_size_1level(%N: index, %ts: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %trip = arith.ceildivui %N, %ts : index
  scf.for %i0 = %c0 to %trip step %c1 {
    // CHECK: ktdf.tiling.derive_size [%{{.*}} : %{{.*}}], total_size = %{{.*}} : index
    %x1 = ktdf.tiling.derive_size [%i0 : %ts], total_size = %N : index
  }
  return
}

// Round-trip: 2-level, dynamic operands at both levels.
// CHECK-LABEL: func.func @tiling.derive_size_2levels
func.func @tiling.derive_size_2levels(%N: index, %ts0: index, %ts1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %trip0 = arith.ceildivui %N, %ts0 : index
  scf.for %i0 = %c0 to %trip0 step %c1 {
    %x1 = ktdf.tiling.derive_size [%i0 : %ts0], total_size = %N : index
    scf.for %i1 = %c0 to %x1 step %c1 {
      // CHECK: ktdf.tiling.derive_size [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}], total_size = %{{.*}} : index
      %x2 = ktdf.tiling.derive_size [%i0 : %ts0], [%i1 : %ts1], total_size = %N : index
    }
  }
  return
}
