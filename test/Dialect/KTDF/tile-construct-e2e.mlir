// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s
// End-to-end: 2-level tiling of a 23-iteration loop with tile sizes 5 and 3.

// CHECK-LABEL: func.func @tile_23_5_3
func.func @tile_23_5_3() {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c3  = arith.constant 3 : index
  %c5  = arith.constant 5 : index
  %c23 = arith.constant 23 : index
  %x0  = arith.ceildivui %c23, %c5 : index

  scf.for %i0 = %c0 to %x0 step %c1 {
    // CHECK: ktdf.tiling.derive_size [%{{.*}} : %{{.*}}], total_size = %{{.*}} : index
    %x1 = ktdf.tiling.derive_size [%i0 : %c5], total_size = %c23 : index
    scf.for %i1 = %c0 to %x1 step %c1 {
      // CHECK: ktdf.tiling.derive_size [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}], total_size = %{{.*}} : index
      %x2 = ktdf.tiling.derive_size [%i0 : %c5], [%i1 : %c3], total_size = %c23 : index
      scf.for %i2 = %c0 to %x2 step %c1 {
        // CHECK: ktdf.tiling.linearize_index [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}], [%{{.*}} : %{{.*}}] : index
        %idx = ktdf.tiling.linearize_index [%i0 : %c5], [%i1 : %c3], [%i2 : %c1] : index
      }
    }
  }
  return
}
