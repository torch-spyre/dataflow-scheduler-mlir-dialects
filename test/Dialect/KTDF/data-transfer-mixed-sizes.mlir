// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Test data transfer with constant sizes (new syntax matching documentation)
// CHECK-LABEL: func.func @test_constant_sizes
func.func @test_constant_sizes() {
  %c0 = arith.constant 0 : index
  
  %A = memref.alloc() : memref<1x64xf16, "DDR">
  %l1_A = memref.alloc() : memref<64xf16, "L1">
  
  // Using constant sizes directly in the IR (matches documentation examples)
  // CHECK: ktdf.data_transfer from {{.*}} size [1, 64] to {{.*}} size [64]
  ktdf.data_transfer from %A[%c0, %c0] size [1, 64] to %l1_A[%c0] size [64] 
    : memref<1x64xf16, "DDR">, memref<64xf16, "L1">
  
  return
}

// Test data transfer with SSA operand sizes (existing syntax - still supported)
// CHECK-LABEL: func.func @test_dynamic_sizes
func.func @test_dynamic_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  
  %A = memref.alloc() : memref<1x64xf16, "DDR">
  %l1_A = memref.alloc() : memref<64xf16, "L1">
  
  // Using SSA values for sizes (existing behavior preserved)
  // CHECK: ktdf.data_transfer from {{.*}} size [{{.*}}, {{.*}}] to {{.*}} size [{{.*}}]
  ktdf.data_transfer from %A[%c0, %c0] size [%c1, %c64] to %l1_A[%c0] size [%c64]
    : memref<1x64xf16, "DDR">, memref<64xf16, "L1">
  
  return
}

// Test data transfer with mixed constant and dynamic sizes
// CHECK-LABEL: func.func @test_mixed_sizes
func.func @test_mixed_sizes(%dynamic_size: index) {
  %c0 = arith.constant 0 : index
  
  %A = memref.alloc() : memref<1x64xf16, "DDR">
  %l1_A = memref.alloc() : memref<64xf16, "L1">
  
  // Mix of constant (1) and dynamic (%dynamic_size) - best of both worlds
  // CHECK: ktdf.data_transfer from {{.*}} size [1, {{.*}}] to {{.*}} size [{{.*}}]
  ktdf.data_transfer from %A[%c0, %c0] size [1, %dynamic_size] to %l1_A[%c0] size [%dynamic_size]
    : memref<1x64xf16, "DDR">, memref<64xf16, "L1">
  
  return
}

// Test FIFO transfers with constant sizes
// CHECK-LABEL: func.func @test_fifo_constant_sizes
func.func @test_fifo_constant_sizes() {
  %c0 = arith.constant 0 : index
  
  %l1_A = memref.alloc() : memref<64xf16, "L1">
  %fifo_slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"l1lu-to-sfu", 64xf16>
  
  // FIFO transfer with constant size
  // CHECK: ktdf.data_transfer from {{.*}} size [64] to {{.*}} size [64]
  ktdf.data_transfer from %l1_A[%c0] size [64] to %fifo_slot size [64]
    : memref<64xf16, "L1">, !ktdf.fifo.slot<"l1lu-to-sfu", 64xf16>
  
  return
}