// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Round-trip: 3-pair subscript via 2-level tiling, dynamic operands.
// CHECK-LABEL: func.func @tiling_linearize_index_2levels
func.func @tiling_linearize_index_2levels(%N: index, %ts0: index, %ts1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %trip0 = arith.ceildivui %N, %ts0 : index
  scf.for %i0 = %c0 to %trip0 step %c1 {
    %x1 = ktdf.tiling.derive_size [%i0 : %ts0], total_size = %N : index
    scf.for %i1 = %c0 to %x1 step %c1 {
      %x2 = ktdf.tiling.derive_size [%i0 : %ts0], [%i1 : %ts1], total_size = %N : index
      scf.for %i2 = %c0 to %x2 step %c1 {
        // CHECK: ktdf.tiling.linearize_index [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}] : index
        %idx = ktdf.tiling.linearize_index [%i0 : %ts0], [%i1 : %ts1], [%i2 : %c1] : index
      }
    }
  }
  return
}

// Round-trip: 2-pair subscript via 1-level tiling, dynamic operands.
// CHECK-LABEL: func.func @tiling_linearize_index_1level
func.func @tiling_linearize_index_1level(%N: index, %ts: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %trip = arith.ceildivui %N, %ts : index
  scf.for %i0 = %c0 to %trip step %c1 {
    %x1 = ktdf.tiling.derive_size [%i0 : %ts], total_size = %N : index
    scf.for %i1 = %c0 to %x1 step %c1 {
      // CHECK: ktdf.tiling.linearize_index [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}] : index
      %idx = ktdf.tiling.linearize_index [%i0 : %ts], [%i1 : %c1] : index
    }
  }
  return
}
