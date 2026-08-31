// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL: func.func @test_ind_data_transfer_gather_to_fifo
func.func @test_ind_data_transfer_gather_to_fifo() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iab  = memref.alloc() : memref<32xindex, "ct_iab">
  %data = memref.alloc() : memref<64x2x64xf16, "global">
  %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>

  // CHECK: ktdf.ind_data_transfer
  // CHECK-NEXT: ind_src = %{{.*}}[%{{.*}}]
  // CHECK-NEXT: dir_src = %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] size [1, 2, 64]
  // CHECK-NEXT: ind_dst = none
  // CHECK-NEXT: dir_dst = %{{.*}} size [2, 64]
  // CHECK-NEXT: : memref<32xindex, "ct_iab">, memref<64x2x64xf16, "global">, none, !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>
  ktdf.ind_data_transfer
      ind_src = %iab[%c1]
      dir_src = %data[%c0, %c0, %c0] size [1, 2, 64]
      ind_dst = none
      dir_dst = %slot                 size [2, 64]
      : memref<32xindex, "ct_iab">,
        memref<64x2x64xf16, "global">,
        none,
        !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>
  return
}

// CHECK-LABEL: func.func @test_ind_data_transfer_gather_to_memref
func.func @test_ind_data_transfer_gather_to_memref() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iab     = memref.alloc() : memref<32xindex, "ct_iab">
  %data    = memref.alloc() : memref<64x2x64xf16, "global">
  %staging = memref.alloc() : memref<2x64xf16, "ct_local">

  // CHECK: ktdf.ind_data_transfer
  // CHECK-NEXT: ind_src = %{{.*}}[%{{.*}}]
  // CHECK-NEXT: dir_src = %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] size [1, 2, 64]
  // CHECK-NEXT: ind_dst = none
  // CHECK-NEXT: dir_dst = %{{.*}}[%{{.*}}, %{{.*}}] size [2, 64]
  // CHECK-NEXT: : memref<32xindex, "ct_iab">, memref<64x2x64xf16, "global">, none, memref<2x64xf16, "ct_local">
  ktdf.ind_data_transfer
      ind_src = %iab[%c1]
      dir_src = %data[%c0, %c0, %c0]    size [1, 2, 64]
      ind_dst = none
      dir_dst = %staging[%c0, %c0]      size [2, 64]
      : memref<32xindex, "ct_iab">,
        memref<64x2x64xf16, "global">,
        none,
        memref<2x64xf16, "ct_local">
  return
}

// CHECK-LABEL: func.func @test_ind_data_transfer_scatter
func.func @test_ind_data_transfer_scatter() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iab     = memref.alloc() : memref<32xindex, "ct_iab">
  %staging = memref.alloc() : memref<2x64xf16, "ct_local">
  %dst     = memref.alloc() : memref<64x2x64xf16, "global">

  // CHECK: ktdf.ind_data_transfer
  // CHECK-NEXT: ind_src = none
  // CHECK-NEXT: dir_src = %{{.*}}[%{{.*}}, %{{.*}}] size [2, 64]
  // CHECK-NEXT: ind_dst = %{{.*}}[%{{.*}}]
  // CHECK-NEXT: dir_dst = %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] size [1, 2, 64]
  // CHECK-NEXT: : none, memref<2x64xf16, "ct_local">, memref<32xindex, "ct_iab">, memref<64x2x64xf16, "global">
  ktdf.ind_data_transfer
      ind_src = none
      dir_src = %staging[%c0, %c0]      size [2, 64]
      ind_dst = %iab[%c1]
      dir_dst = %dst[%c0, %c0, %c0]     size [1, 2, 64]
      : none,
        memref<2x64xf16, "ct_local">,
        memref<32xindex, "ct_iab">,
        memref<64x2x64xf16, "global">
  return
}

// CHECK-LABEL: func.func @test_ind_data_transfer_in_pipeline
func.func @test_ind_data_transfer_in_pipeline() {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %iab  = memref.alloc() : memref<32xindex, "ct_iab">
  %data = memref.alloc() : memref<64x2x64xf16, "global">

  ktdf.pipeline {
    %prv:2 = ktdf.private -> (
        !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>,
        !ktdf.token
    ) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>
      %tok  = ktdf.create_token : !ktdf.token
      ktdf.private_yield %slot, %tok
          : !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>, !ktdf.token
    }

    // CHECK: ktdf.stage
    ktdf.stage depends_in(none) depends_out(%prv#1) {
      // CHECK: ktdf.ind_data_transfer
      ktdf.ind_data_transfer
          ind_src = %iab[%c1]
          dir_src = %data[%c0, %c0, %c0] size [1, 2, 64]
          ind_dst = none
          dir_dst = %prv#0               size [2, 64]
          : memref<32xindex, "ct_iab">,
            memref<64x2x64xf16, "global">,
            none,
            !ktdf.fifo.slot<"ct_load" -> "ct_compute", 128xf16>
    }
  }
  return
}
